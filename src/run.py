import os
import json
from collections import defaultdict
from time import time

from src.agents import get_agents
from src.mcts import MCTSNode, default_mcts_selection, beam_search, progressive_widening, progressive_widening_all, \
    ucb1_recursive
from src.dataset import get_datasets_fpaths, get_load_dataset_experiment
from src.logger import TreeLogger

from src.beliefs import calculate_prior_and_posterior_beliefs
from datetime import datetime
import shutil

from src.args import ArgParser
from src.mcts_utils import load_mcts_from_json, save_nodes, get_msgs_from_latest_query, setup_group_chat, \
    print_node_info, get_self_value, get_context_string


def select_node(selection_method, root, nodes_by_level, n_warmstart=0):
    """
    Select the next node to expand in MCTS using the provided selection method.

    Args:
        selection_method: Function to select nodes in MCTS.
        root: Root MCTSNode to select from.
        nodes_by_level: Dictionary of nodes by level.
        n_warmstart: Number of warmstart experiments to run after data loading but before MCTS selection.

    Returns:
        Selected MCTSNode for expansion.
    """
    n_children_at_data_loader = len(nodes_by_level[2])

    # If there are warmstart experiments left to run, select the data loader node to execute the next experiment.
    if len(nodes_by_level[1]) > 0 and (n_warmstart - n_children_at_data_loader) > 0:
        return nodes_by_level[1][0]

    return selection_method(root, nodes_by_level)


def compute_and_store_reward(node, belief_model_name, belief_temperature, reasoning_effort,
                             n_belief_samples, implicit_bayes_posterior, surprisal_width, belief_mode,
                             use_binary_reward, all_surprisals=None, use_online_beliefs=False,
                             evidence_weight=1.0, kl_scale=20.0, reward_mode="belief", TEMP_LOG=None):
    s_conditioned_prior = None
    evidence_msg = []

    # If there are past surprisal, computed the s-conditioned prior
    if all_surprisals is not None and len(all_surprisals) > 0 and use_online_beliefs:
        # Build evidence message for prior belief elicitation
        evidence_msg = [{
            "role": "user",
            "content": "Previous study:\n\n" + get_context_string(
                hyp_exp_query=f"Hypothesis: {nodes_by_level[level_index[0]][level_index[1]].hypothesis}",
                analysis=nodes_by_level[level_index[0]][level_index[1]].analysis,
                review=nodes_by_level[level_index[0]][level_index[1]].review,
                belief_mean=nodes_by_level[level_index[0]][level_index[1]].posterior.mean,
                include_code_output=False
            )
        } for level_index in all_surprisals]
        try:
            pt_prior, s_conditioned_prior, _, _ = calculate_prior_and_posterior_beliefs(
                node,
                model=belief_model_name,
                temperature=belief_temperature,
                reasoning_effort=reasoning_effort,
                n_samples=n_belief_samples,
                implicit_bayes_posterior=implicit_bayes_posterior,
                surprisal_width=surprisal_width,
                belief_mode=belief_mode,
                evidence_msg=evidence_msg
            )
        except ValueError as e:
            print(f"Error for node {node.id}: {e}")
            node.success = False
            return

        # TEMPORARY LOGGING
        if TEMP_LOG is not None:
            TEMP_LOG.append({
                'node_id': node.id,
                'belief_change': None,
                'kl_divergence': None,
                'hypothesis': node.hypothesis,
                'pt_prior': pt_prior.to_dict(),
                'surprisal_evidence': [e['content'] for e in evidence_msg],
                's_conditioned_prior': s_conditioned_prior.to_dict(),
            })

    # Build the evidence message for the current node
    evidence_msg.append({
        "role": "user",
        "content": "Current experiment:\n\n" + get_context_string(
            hyp_exp_query=node.query,
            code_output=node.code_output,
            analysis=node.analysis,
            review=node.review,
            include_code_output=False
        )
    })

    # Compute the prior and posterior beliefs for the current node
    try:
        prior, posterior, belief_change, kl_divergence = calculate_prior_and_posterior_beliefs(
            node,
            model=belief_model_name,
            temperature=belief_temperature,
            reasoning_effort=reasoning_effort,
            n_samples=n_belief_samples,
            implicit_bayes_posterior=implicit_bayes_posterior,
            surprisal_width=surprisal_width,
            belief_mode=belief_mode,
            prior=s_conditioned_prior,
            evidence_msg=evidence_msg,
            evidence_weight=evidence_weight
        )
    except ValueError as e:
        print(f"Error for node {node.id}: {e}")
        node.success = False
        return

    # TEMPORARY LOGGING
    if TEMP_LOG is not None and len(TEMP_LOG) > 0:
        # Generate the posterior without surprisals
        _, _posterior, _belief_change, _kl_divergence = calculate_prior_and_posterior_beliefs(
            node,
            model=belief_model_name,
            temperature=belief_temperature,
            reasoning_effort=reasoning_effort,
            n_samples=n_belief_samples,
            implicit_bayes_posterior=implicit_bayes_posterior,
            surprisal_width=surprisal_width,
            belief_mode=belief_mode,
            prior=pt_prior,
            evidence_msg=evidence_msg[-1:],
            evidence_weight=evidence_weight
        )

        TEMP_LOG[-1]['current_evidence'] = evidence_msg[-1]['content']
        TEMP_LOG[-1]['online_posterior'] = posterior.to_dict()
        TEMP_LOG[-1]['belief_change'] = belief_change
        TEMP_LOG[-1]['kl_divergence'] = kl_divergence
        TEMP_LOG[-1]['offline_posterior'] = _posterior.to_dict()
        TEMP_LOG[-1]['offline_belief_change'] = _belief_change
        TEMP_LOG[-1]['offline_kl_divergence'] = _kl_divergence
        TEMP_LOG[-1]['current_surprisals'] = all_surprisals.copy()

        print(f"\n\n======================= SURPRISAL-CONDITION BELIEFS =======================\n")
        print(json.dumps({k: v for k, v in TEMP_LOG[-1].items() if
                          k in ["pt_prior", "s_conditioned_prior", "online_posterior", "offline_posterior"]}, indent=2))

    node.prior = prior
    node.posterior = posterior
    node.belief_change = belief_change
    node.kl_divergence = kl_divergence
    # Compute reward and surprisal
    node.self_value, node.surprising = get_self_value(belief_change=node.belief_change,
                                                      kl_divergence=node.kl_divergence,
                                                      binary=use_binary_reward, width=surprisal_width,
                                                      kl_scale=kl_scale, mode=reward_mode)
    if node.surprising:
        # Store the surprisal
        all_surprisals.append((node.level, node.node_idx))
        # TODO: Update all past nodes with the new surprisal set


def run_mcts(
        root,
        nodes_by_level,
        dataset_paths,
        log_dirname,
        work_dir,
        model_name="gpt-4o",
        belief_model_name="gpt-4o",
        max_iterations=100,
        branching_factor=8,
        max_rounds=100000,
        selection_method=None,
        allow_generate_experiments=False,
        n_belief_samples=30,
        k_parents=3,
        temperature=1.0,
        belief_temperature=1.0,
        reasoning_effort="medium",
        implicit_bayes_posterior=False,
        surprisal_width=0.2,
        user_query=None,
        belief_mode="categorical",
        use_binary_reward=True,
        run_dedupe=True,
        experiment_first=False,
        code_timeout=30 * 60,
        n_warmstart=0,
        use_online_beliefs=False,
        evidence_weight=1.0,
        kl_scale=20.0,
        reward_mode="belief_and_kl",
        warmstart_experiments=None
):
    """
    Run AutoDS exploration. In MCTS, root node level=0 is a dummy node with no experiment, level=1 is the first real node with the dataset loading experiment, levels > 1 are the actual MCTS nodes with hypotheses and experiments.

    Args:
        root: Root MCTSNode to continue from.
        nodes_by_level: Dictionary to store nodes by level.
        dataset_paths: List of paths to dataset files.
        log_dirname: Directory to save logs and MCTS nodes.
        work_dir: Working directory for agents.
        model_name: LLM model name for agents.
        belief_model_name: LLM model name for belief distribution agent.
        max_iterations: Maximum number of MCTS iterations.
        branching_factor: Maximum number of children per node.
        max_rounds: Maximum number of rounds for the group chat.
        selection_method: Function to select nodes in MCTS (default is UCB1).
        allow_generate_experiments: Whether to allow nodes to generate new experiments on demand.
        n_belief_samples: Number of samples for belief distribution evaluation.
        k_parents: Number of parent levels to include in logs (None for all).
        temperature: Temperature setting for all agents (except belief agent).
        belief_temperature: Temperature setting for the belief agent.
        reasoning_effort: Reasoning effort for OpenAI o-series models.
        implicit_bayes_posterior: Whether to use the belief samples with evidence as the direct posterior or to use a Bayesian update that explicitly combines it with the prior.
        surprisal_width: Minimum difference in mean prior and posterior probabilities required to count as a surprisal.
        user_query: Custom user query to condition experiment generation during exploration.
        belief_mode: Belief elicitation mode (boolean, categorical, categorical_numeric, or probability).
        use_binary_reward: Whether to use binary reward for MCTS instead of a continuous reward (belief change).
        run_dedupe: Whether to deduplicate nodes before saving to JSON and CSV.
        experiment_first: If True, an experiment will be generated before its hypothesis.
        code_timeout: Timeout for code execution in seconds (default is 30 minutes).
        n_warmstart: Number of warmstart experiments to run after data loading but before MCTS selection.
        use_online_beliefs: Whether to use online beliefs (i.e., beliefs updated with evidence from previous nodes).
        evidence_weight: Weight for the experimental evidence for posterior calculation.
        kl_scale: Normalization factor for KL divergence in reward calculation.
        reward_mode: Mode for reward calculation (belief, kl, or belief_and_kl).
        warmstart_experiments: Path to JSON file with warmstart experiments to run after data loading but before MCTS selection.
    """
    # Setup logger
    logger = TreeLogger(log_dirname)

    # Track time
    start_time = time()

    # Create work directory if it doesn't exist
    os.makedirs(work_dir, exist_ok=True)

    # Copy the dataset file paths to the working directory (to avoid modifying the original dataset)
    for dataset_fpath in dataset_paths:
        shutil.copy(dataset_fpath, work_dir)

    # Get agents
    agent_objs = get_agents(work_dir, model_name=model_name, temperature=temperature,
                            reasoning_effort=reasoning_effort, branching_factor=branching_factor,
                            user_query=user_query, experiment_first=experiment_first, code_timeout=code_timeout)
    user_proxy = agent_objs["user_proxy"]
    experiment_generator = agent_objs["experiment_generator"]

    # Set up the group chat
    groupchat, chat_manager = setup_group_chat(agent_objs, max_rounds)

    if selection_method is None:
        # Default selection method is UCB1
        selection_method = default_mcts_selection(exploration_weight=1.0)

    # Store the list of (level, node_idx) tuples for surprising nodes; if resuming, load them from the previous run
    all_surprisals = []
    for level in nodes_by_level:
        for node in nodes_by_level[level]:
            if node.surprising:
                all_surprisals.append((node.level, node.node_idx))

    # Load warmstart experiments if provided
    _warmstart_experiments = None
    if warmstart_experiments is not None:
        with open(warmstart_experiments, "r") as f:
            _warmstart_experiments = json.load(f)

    # TEMPORARY LOGGING
    TEMP_LOG = []

    try:
        for iteration_idx in range(max_iterations):
            # MCTS SELECTION, EXPANSION, and EXECUTION
            print(f"\n\n######### ITERATION {iteration_idx + 1} / {max_iterations} #########\n")

            # Select the next node to expand
            node = select_node(selection_method, root, nodes_by_level, n_warmstart)
            # Fetch or generate the next experiment from the selected node (retries built in)
            new_experiment, new_query = node.get_next_experiment(experiment_generator=experiment_generator)

            if new_query is not None:
                # Create a new node for the next experiment
                new_level = node.level + 1
                new_node_idx = len(nodes_by_level[new_level])
                node = MCTSNode(level=new_level, node_idx=new_node_idx, hypothesis=new_experiment["hypothesis"],
                                experiment_plan=new_experiment["experiment_plan"], query=new_query, parent=node,
                                allow_generate_experiments=allow_generate_experiments and new_level > 0,
                                untried_experiments=_warmstart_experiments if new_level == 1 else None)
                # Update logger state
                logger.level = node.level
                logger.node_idx = node.node_idx

                # Load previous explorations (make sure the root is always included)
                node_context = []
                if node.level > 1:
                    node_context = [root.children[0].get_context(include_code_output=True)] + node.get_path_context(
                        k=k_parents - 1, skip_root=True)
                node_messages = []
                if node_context is not None:
                    node_messages += [
                        {"name": "user_proxy", "role": "user", "content": "PREVIOUS EXPLORATION:\n\n" + n} for n in
                        node_context]
                node_messages += [
                    {"name": "user_proxy", "role": "user", "content": node.query}]
                _, last_message = chat_manager.resume(messages=node_messages)

                # Track time per node
                _node_start_time = time()

                # Execute current experiment and generate new experiments
                user_proxy.initiate_chat(recipient=chat_manager, message=last_message, clear_history=False)

                # Store the raw message logs for the node
                logger.log_node(node.level, node.node_idx, chat_manager.messages_to_string(groupchat.messages))

                # Get messages starting from the current query and update the node
                node.messages = get_msgs_from_latest_query(groupchat.messages)
                node.read_experiment_from_messages(
                    store_new_experiments=False if node.level == 1 and _warmstart_experiments is not None else True)
                # Calculate beliefs and rewards
                if node.success and node.level > 1:
                    compute_and_store_reward(node, belief_model_name, belief_temperature, reasoning_effort,
                                             n_belief_samples, implicit_bayes_posterior, surprisal_width, belief_mode,
                                             use_binary_reward, all_surprisals, use_online_beliefs=use_online_beliefs,
                                             evidence_weight=evidence_weight, kl_scale=kl_scale,
                                             reward_mode=reward_mode, TEMP_LOG=TEMP_LOG)

                    if node.success:  # i.e., reward was computed successfully
                        # Print debug information
                        print_node_info(node)

                        # TEMPORARY LOGGING
                        if TEMP_LOG:
                            temp_log_file = os.path.join(log_dirname, "temp_log.json")
                            with open(temp_log_file, "w") as f:
                                json.dump(TEMP_LOG, f, indent=2)
                            print(f"Temporary log saved to {temp_log_file}")

                # End time tracking for the node
                _node_end_time = time()
                node.time_elapsed = round(_node_end_time - _node_start_time, 2)

                # Add the new node to the nodes_by_level dictionary
                nodes_by_level[node.level].append(node)

                # MCTS BACKPROPAGATION
                node.update_counts(visits=1, reward=node.self_value)

                # Save the current state of the node
                node_file = os.path.join(log_dirname, f"mcts_{node.id}.json")
                with open(node_file, "w") as f:
                    json.dump(node.to_dict(), f, indent=2)
            else:
                # No new experiment was generated; don't change the state of the tree and sample again
                print(f"No new experiment generated for node {node.level}_{node.node_idx}. Skipping this iteration.")
    except KeyboardInterrupt:
        print("\n\n######### EXPLORATION INTERRUPTED! SAVING THE CURRENT STATE... #########\n\n")

    # End time tracking
    end_time = time()
    time_elapsed = end_time - start_time

    # Save all MCTS nodes
    save_nodes(nodes_by_level, log_dirname, run_dedupe, belief_model_name, time_elapsed=time_elapsed)


if __name__ == "__main__":
    parser = ArgParser()
    args = parser.parse_args()
    print("Script arguments:")
    print(args.__dict__, "\n")

    # Validate and fix arguments
    if "o4-mini" in args.model and args.temperature is not None:
        print("Warning: Setting temperature for o4-mini is not permitted. Using default None.")
        args.temperature = None
    if "o4-mini" in args.belief_model and args.belief_temperature is not None:
        print("Warning: Setting temperature for o4-mini belief model is not permitted. Using default None.")
        args.belief_temperature = None

    # Create log directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dirname = os.path.join(args.out_dir, timestamp) if args.timestamp_dir else args.out_dir
    work_dirname = os.path.join(args.work_dir, timestamp) if args.timestamp_dir else args.work_dir

    # Setup logger
    logger = TreeLogger(log_dirname)

    # Save args
    args_file = os.path.join(log_dirname, "args.json")
    with open(args_file, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"\nArguments saved to {args_file}\n")

    # Get dataset paths
    dataset_paths, dataset_metadata = get_datasets_fpaths(args.dataset_metadata,
                                                          is_blade=args.dataset_metadata_type == 'blade')
    load_dataset_experiment = get_load_dataset_experiment(dataset_paths, dataset_metadata, run_eda=args.run_eda,
                                                          dataset_metadata_type=args.dataset_metadata_type)

    if args.continue_from_dir or args.continue_from_json:
        if args.continue_from_dir is not None:
            # Load nodes from a directory
            root, nodes_by_level = load_mcts_from_json(args.continue_from_dir, args)
            # Copy all files except args.json from continue_from_dir to the new log directory
            for filename in os.listdir(args.continue_from_dir):
                if filename != "args.json":
                    shutil.copy(os.path.join(args.continue_from_dir, filename), os.path.join(log_dirname, filename))
        else:
            # Load from a JSON file that contains all the nodes (not de-duplicated)
            root, nodes_by_level = load_mcts_from_json(args.continue_from_json, args)

        if args.only_save_results:
            # Save nodes to JSON and exit
            save_nodes(nodes_by_level, log_dirname, run_dedupe=args.dedupe, model=args.belief_model)
            exit(0)

        if args.continue_from_dir is not None:
            # Copy all files except args.json from continue_from_dir to the new log directory
            for filename in os.listdir(args.continue_from_dir):
                if filename != "args.json":
                    shutil.copy(os.path.join(args.continue_from_dir, filename), os.path.join(log_dirname, filename))
        else:
            # Create the individual node files in the log directory
            for node in nodes_by_level.values():
                for n in node:
                    node_file = os.path.join(log_dirname, f"mcts_{n.id}.json")
                    with open(node_file, "w") as f:
                        json.dump(n.to_dict(), f, indent=2)

        # Calculate remaining iterations to reach n_experiments
        total_nodes = sum(len(nodes) for nodes in nodes_by_level.values())
        remaining_iters = (args.n_experiments + 1) - total_nodes  # + 1 to account for root node
        if remaining_iters <= 0:
            print(f"Already reached or exceeded target of {args.n_experiments} experiments")
            exit(0)
        print(
            f"RESUMING: Running {remaining_iters} more experiments to reach the target experiment count of {args.n_experiments}.\n")
    else:
        root = MCTSNode(level=0, node_idx=0, hypothesis=None, query=None,
                        allow_generate_experiments=False, untried_experiments=[load_dataset_experiment])
        nodes_by_level = defaultdict(list)
        nodes_by_level[0].append(root)
        remaining_iters = args.n_experiments + 1  # + 1 to account for root node

    # Set up selection method based on args
    if args.mcts_selection == "pw":
        # Progressive Widening
        assert args.pw_k is not None and args.pw_alpha is not None
        selection_method = progressive_widening(args.pw_k, args.pw_alpha, args.exploration_weight)
    elif args.mcts_selection == "pw_all":
        # Progressive Widening
        assert args.pw_k is not None and args.pw_alpha is not None
        selection_method = progressive_widening_all(args.pw_k, args.pw_alpha, args.exploration_weight)
    elif args.mcts_selection == "beam_search":
        # Beam Search
        selection_method = beam_search(args.k_experiments, args.beam_width, args.out_dir)
    elif args.mcts_selection == "ucb1":
        # UCB1
        selection_method = default_mcts_selection(args.exploration_weight)
    elif args.mcts_selection == "ucb1_recursive":
        # UCB1 recursive
        selection_method = ucb1_recursive(args.exploration_weight)
    else:
        raise ValueError(f"Unknown MCTS selection method: {args.mcts_selection}")
    print(f"MCTS selection method: {args.mcts_selection}\n")

    # Run exploration
    run_mcts(
        root=root,
        nodes_by_level=nodes_by_level,
        dataset_paths=dataset_paths,
        log_dirname=log_dirname,
        work_dir=work_dirname,
        max_iterations=remaining_iters,
        branching_factor=args.k_experiments,
        selection_method=selection_method,
        allow_generate_experiments=args.allow_generate_experiments,
        n_belief_samples=args.n_belief_samples,
        k_parents=args.k_parents,
        model_name=args.model,
        belief_model_name=args.belief_model,
        temperature=args.temperature,
        belief_temperature=args.belief_temperature,
        reasoning_effort=args.reasoning_effort,
        implicit_bayes_posterior=args.implicit_bayes_posterior,
        surprisal_width=args.surprisal_width,
        user_query=args.user_query,
        belief_mode=args.belief_mode,
        use_binary_reward=args.use_binary_reward,
        run_dedupe=args.dedupe,
        experiment_first=args.experiment_first,
        code_timeout=args.code_timeout,
        n_warmstart=args.n_warmstart,
        use_online_beliefs=args.use_online_beliefs,
        evidence_weight=args.evidence_weight,
        kl_scale=args.kl_scale,
        reward_mode=args.reward_mode,
        warmstart_experiments=args.warmstart_experiments,
    )

    if args.delete_work_dir:
        shutil.rmtree(args.work_dir)
        print(f"\nDELETED WORKING DIRECTORY: {args.work_dir}")

    print(f"\nRUN FINISHED!\n\nLOGS: {log_dirname}")

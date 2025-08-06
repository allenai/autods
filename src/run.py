import os
import json
from collections import defaultdict

from src.agents import get_agents
from src.mcts import MCTSNode, default_mcts_selection, beam_search, progressive_widening
from src.dataset import get_dataset_description, get_blade_description, get_datasets_fpaths
from src.logger import TreeLogger

from src.beliefs import calculate_prior_and_posterior_beliefs
from datetime import datetime
import shutil
from openai import BadRequestError as OpenAIBadRequestError

from src.args import ArgParser
from src.mcts_utils import load_mcts_from_json, save_nodes, get_msgs_from_latest_query, setup_group_chat, \
    print_node_info
from src.utils import try_loading_dict


def run_mcts(
        query,
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
        root=None,
        nodes_by_level=None,
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
        experiment_first=False
):
    """
    Run AutoDS exploration. In MCTS, root node level=0 is a dummy node with no experiment, level=1 is the first real node with the dataset loading experiment, levels > 1 are the actual MCTS nodes with hypotheses and experiments.

    Args:
        query: Initial query to start the exploration.
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
        root: Optional root MCTSNode to continue from. If None, initializes a new root.
        nodes_by_level: Optional dictionary to store nodes by level. If None, initializes a new one.
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
    """
    # Setup logger
    logger = TreeLogger(log_dirname)

    # Create work directory if it doesn't exist
    os.makedirs(work_dir, exist_ok=True)

    # Copy the dataset file paths to the working directory (to avoid modifying the original dataset)
    for dataset_fpath in dataset_paths:
        shutil.copy(dataset_fpath, work_dir)

    # Get the structured agents
    structured_agents = get_agents(work_dir, model_name=model_name, temperature=temperature,
                                   reasoning_effort=reasoning_effort, branching_factor=branching_factor,
                                   user_query=user_query, experiment_first=experiment_first)
    agent_objs = {agent.name: agent for agent in structured_agents}
    user_proxy = agent_objs["user_proxy"]

    # Set up the group chat
    groupchat, chat_manager = setup_group_chat(agent_objs, max_rounds)

    if selection_method is None:
        # Default selection method is UCB1
        selection_method = default_mcts_selection(exploration_weight=1.0)

    # Initialize the root node if not continuing from saved state
    if root is None:
        root = MCTSNode(level=0, node_idx=0, hypothesis=None, query=None, untried_experiments=[query],
                        allow_generate_experiments=False)
        nodes_by_level = defaultdict(list)
        nodes_by_level[0].append(root)

    experiment_generator_agent = agent_objs["experiment_generator"]

    try:
        for iteration_idx in range(max_iterations):
            # MCTS SELECTION, EXPANSION, and EXECUTION
            print(f"\n\n######### ITERATION {iteration_idx + 1} / {max_iterations} #########\n")
            node = selection_method(root)

            exp, new_hypothesis, new_query = node.get_next_experiment(
                experiment_generator_agent=experiment_generator_agent)

            if new_query is not None:
                try:
                    new_level = node.level + 1
                    new_node_idx = len(nodes_by_level[new_level])
                    allow_generate = allow_generate_experiments if new_level > 0 else False
                    child = MCTSNode(level=new_level, node_idx=new_node_idx, hypothesis=new_hypothesis, query=new_query,
                                     parent=node, allow_generate_experiments=allow_generate)
                    node.children.append(child)
                    nodes_by_level[new_level].append(child)
                    node = child

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

                    # Generate new experiments
                    user_proxy.initiate_chat(recipient=chat_manager, message=last_message, clear_history=False)
                    logger.log_node(node.level, node.node_idx, chat_manager.messages_to_string(groupchat.messages))
                    node.messages = get_msgs_from_latest_query(groupchat.messages)

                    # Store generated experiments
                    assert node.messages[-1]['name'] == "experiment_generator"
                    experiments = try_loading_dict(node.messages[-1]['content']).get('experiments', [])
                    node.untried_experiments += experiments

                    # Calculate beliefs and rewards
                    if node.successful and node.level > 1:
                        # Belief elicitation
                        is_surprisal, belief_change, prior, posterior = calculate_prior_and_posterior_beliefs(
                            node,
                            model=belief_model_name,
                            temperature=belief_temperature,
                            reasoning_effort=reasoning_effort,
                            n_samples=n_belief_samples,
                            implicit_bayes_posterior=implicit_bayes_posterior,
                            surprisal_width=surprisal_width,
                            belief_mode=belief_mode
                        )
                        node.surprising = is_surprisal
                        node.prior = prior
                        node.posterior = posterior
                        node.belief_change = belief_change

                        # Reward based on surprising hypothesis
                        node.self_value = (1 if node.surprising else 0) if use_binary_reward else (
                            node.belief_change if node.belief_change else 0)

                        # Print debug information
                        print_node_info(node)

                except OpenAIBadRequestError:
                    print(f"BadRequestError on iteration {iteration_idx} for node {node.level}_{node.node_idx}. "
                          "Skipping this node and continuing.")
                    continue

            # MCTS BACKPROPAGATION
            node.update_counts(visits=1, reward=node.self_value)

            # Save the individual node after backpropagation
            node_file = os.path.join(log_dirname, f"mcts_{node.id}.json")
            with open(node_file, "w") as f:
                json.dump(node.to_dict(), f, indent=2)

    except KeyboardInterrupt:
        print("\n\n######### EXPLORATION INTERRUPTED! SAVING THE CURRENT STATE... #########\n\n")

    # Save all MCTS nodes
    save_nodes(nodes_by_level, log_dirname, run_dedupe, belief_model_name)


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
    dataset_paths = get_datasets_fpaths(args.dataset_metadata, is_blade=args.dataset_metadata_type == 'blade')

    if args.continue_from_dir or args.continue_from_json:
        if args.continue_from_dir is not None:
            # Load nodes from directory
            root, nodes_by_level = load_mcts_from_json(args.continue_from_dir, args)
            # Copy all files except args.json from continue_from_dir to the new log directory
            for filename in os.listdir(args.continue_from_dir):
                if filename != "args.json":
                    shutil.copy(os.path.join(args.continue_from_dir, filename), os.path.join(log_dirname, filename))
        else:
            # Load from collected JSON file
            root, nodes_by_level = load_mcts_from_json(args.continue_from_json, args)

        if args.only_save_results:
            # Save nodes to JSON and exit
            save_nodes(nodes_by_level, log_dirname, run_dedupe=args.dedupe, model=args.belief_model)
            exit(0)

        # Load MCTS state
        query = root.children[0].query if root.children else None

        # Calculate remaining iterations to reach n_experiments
        total_nodes = sum(len(nodes) for nodes in nodes_by_level.values())
        remaining_iters = (args.n_experiments + 1) - total_nodes  # + 1 to account for root node
        if remaining_iters <= 0:
            print(f"Already reached or exceeded target of {args.n_experiments} experiments")
            exit(0)
        print(
            f"RESUMING: Running {remaining_iters} more experiments to reach the target experiment count of {args.n_experiments}.\n")
    else:
        root = None
        nodes_by_level = None
        remaining_iters = args.n_experiments + 1  # + 1 to account for root node
        load_dataset_objective = "Load the dataset and generate summary statistics. "
        load_dataset_steps = f"1. Load the dataset(s) at {[os.path.basename(dp) for dp in dataset_paths]}.\n2. Generate summary statistics for the dataset(s).\n3. Perform some exploratory data analysis (EDA) on the dataset(s) to get a better understanding of the data."
        load_dataset_deliverables = "1. Dataset(s) loaded.\n2. Summary statistics generated.\n3. Exploratory data analysis (EDA) performed."

        if args.dataset_metadata_type == 'blade':
            load_dataset_objective += f"Here is the dataset metadata:\n\n{get_blade_description(args.dataset_metadata)}"
        else:  # DiscoveryBench-style
            load_dataset_objective += f"Here is the dataset metadata:\n\n{get_dataset_description(args.dataset_metadata)}"

        query = {
            "hypothesis": None,
            "experiment_plan": {
                "objective": load_dataset_objective,
                "steps": load_dataset_steps,
                "deliverables": load_dataset_deliverables
            }
        }

    # Set up selection method based on args
    if args.mcts_selection == "pw":
        # Progressive Widening
        assert args.pw_k is not None and args.pw_alpha is not None
        selection_method = progressive_widening(args.pw_k, args.pw_alpha, args.exploration_weight)
    elif args.mcts_selection == "beam_search":
        # Beam Search
        selection_method = beam_search(args.k_experiments, args.beam_width, args.out_dir)
    elif args.mcts_selection == "ucb1":
        # UCB1
        selection_method = default_mcts_selection(args.exploration_weight)
    else:
        raise ValueError(f"Unknown MCTS selection method: {args.mcts_selection}")
    print(f"MCTS selection method: {args.mcts_selection}\n")

    # Run exploration
    run_mcts(
        query=query,
        dataset_paths=dataset_paths,
        log_dirname=log_dirname,
        work_dir=work_dirname,
        max_iterations=remaining_iters,
        branching_factor=args.k_experiments,
        selection_method=selection_method,
        allow_generate_experiments=args.allow_generate_experiments,
        n_belief_samples=args.n_belief_samples,
        root=root,
        nodes_by_level=nodes_by_level,
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
        experiment_first=args.experiment_first
    )

    if args.delete_work_dir:
        shutil.rmtree(args.work_dir)
        print(f"\nDELETED WORKING DIRECTORY: {args.work_dir}")

    print(f"\nRUN FINISHED!\n\nLOGS: {log_dirname}")

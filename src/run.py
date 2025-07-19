import os
import json
from collections import defaultdict
import numpy as np
import autogen
from autogen import GroupChatManager
from agents import get_agents
from nodes_to_csv import nodes_to_csv
from transitions import SpeakerSelector
from dataset import get_dataset_description, get_blade_description, get_ai2_description, get_datasets_fpaths
from logger import TreeLogger
from structured_outputs import Hypothesis
from typing import Optional
from beliefs import calculate_prior_and_posterior_beliefs, BELIEF_MODE_TO_CLS
from datetime import datetime
import random
import shutil
from openai import BadRequestError as OpenAIBadRequestError

from reconstruct_mcts import collect_logs_for_resumption
from args import ArgParser


def default_mcts_selection(exploration_weight):
    def select(node):
        # Traverse the tree until we find a node with untried experiments
        while node.children and not node.has_untried_experiments():
            # Select the child with the highest UCB1 value
            node = max(node.children, key=lambda n: ucb1(n, exploration_weight))
        return node

    return select


def progressive_widening(k, alpha, exploration_weight=1.0):
    """
    Create a progressive widening selection function.

    Args:
        k: Progressive widening constant.
        alpha: Progressive widening exponent.

    Returns:
        A callable function that accepts a `node` and returns the selected node.
    """

    def select(node):
        # Get the number of visits and children for the current node
        num_visits = node.visits
        num_children = len(node.children)

        # Check if we can add a new child based on the progressive widening condition
        if (num_children < k * (num_visits ** alpha)) and node.has_untried_experiments():
            # Sample a new child (expand the tree)
            return node

        # Otherwise, recursively sample from the children
        if node.children:
            # Select a child node recursively using the same selection function
            return select(max(node.children, key=lambda n: ucb1(n, exploration_weight)))

        # If no children exist, return the current node
        return node

    return select


def beam_search(branching_factor, beam_width, log_dirname=None):
    """
    Create a beam search selection function.

    Args:
        branching_factor: Maximum number of children per node
        beam_width: Number of nodes to keep in beam

    Returns:
        A callable function that accepts a root node and returns selected node
    """
    beam = []  # Current nodes in beam

    def select(root):
        nonlocal beam

        # Initialize beam with root if empty
        if not beam:
            beam = [root]
            # Log initial beam state
            if log_dirname:
                beam_state = [{"level": node.level, "node_idx": node.node_idx} for node in beam]
                with open(os.path.join(log_dirname, f"beam_level_{root.level}.json"), "w") as f:
                    json.dump(beam_state, f, indent=2)

        # Try nodes in current beam
        for node in beam:
            if node.has_untried_experiments() and len(node.children) < branching_factor:
                return node

        # All nodes in beam are exhausted, select new beam
        all_children = []
        for node in beam:
            all_children.extend(node.children)

        # Sort children by UCB1 score and select top beam_width
        if all_children:
            beam = sorted(all_children, key=lambda n: ucb1(n), reverse=True)[:beam_width]
            # Log new beam state
            if log_dirname:
                beam_state = [{"level": node.level, "node_idx": node.node_idx} for node in beam]
                level = beam[0].level if beam else 0
                with open(os.path.join(log_dirname, f"beam_level_{level}.json"), "w") as f:
                    json.dump(beam_state, f, indent=2)
            return select(root)  # Recurse with new beam

        return beam[0]  # Default to first beam node if no children

    return select


class MCTSNode(object):
    _hypothesis: Optional[Hypothesis]
    _creation_counter = 0

    def __init__(self, level, node_idx, parent_idx, query, parent=None, untried_experiments=None,
                 allow_generate_experiments=False):
        self.level = level
        self.node_idx = node_idx
        self.parent_idx = parent_idx
        self.query = query
        self.parent: Optional[MCTSNode] = parent
        self.children = []
        self.visits = 0  # Visits to this node or its children
        self.value = 0  # Number of surprising hypotheses
        self.untried_experiments = untried_experiments  # Will store potential experiments
        self.tried_experiments = []  # Track all tried experiments
        self.allow_generate_experiments = allow_generate_experiments
        self.messages = []  # Store messages for this node

        # Belief
        self.surprising: Optional[bool] = None
        self.prior = None
        self.posterior = None
        self.belief_change: Optional[float] = None  # Change in belief from prior to posterior

        self.creation_index = MCTSNode._creation_counter  # Track creation order
        MCTSNode._creation_counter += 1
        self.timestamp = datetime.now().isoformat()

    @property
    def hypothesis(self):
        if not hasattr(self, '_hypothesis'):
            self._hypothesis = None
            if self.messages:
                for msg in self.messages:
                    if msg.get("name") == "hypothesis_generator":
                        try:
                            self._hypothesis = json.loads(msg["content"])
                        except (json.JSONDecodeError, TypeError):
                            self._hypothesis = Hypothesis(hypothesis="Failed to parse")
        return self._hypothesis

    @property
    def successful(self):
        if not hasattr(self, '_successful'):
            # Find the last experiment_reviewer message
            self._successful = False
            if self.messages:
                for msg in reversed(self.messages):
                    if msg.get("name") == "experiment_reviewer":
                        try:
                            last_review = json.loads(msg["content"])
                            self._successful = last_review.get("success", False)
                        except (json.JSONDecodeError, TypeError):
                            self._successful = False
                        break
        return self._successful

    def get_next_experiment(self, experiment_generator_agent=None):
        """
        Returns the next untried experiment. If none left and allowed, generates more using the experiment generator agent.
        """
        exp, new_query = None, None
        if self.untried_experiments:
            idx = random.randrange(len(self.untried_experiments))
            exp = self.untried_experiments.pop(idx)
            self.tried_experiments.append(exp)
        if self.allow_generate_experiments and experiment_generator_agent is not None:
            # Generate new experiments on demand
            # Provide all previous messages and tried experiments as context
            prompt = {
                "role": "user",
                "content": f"Generate new experiments given the following context. Previously tried experiments: {json.dumps(self.tried_experiments)}"
            }
            messages = self.messages + [prompt]
            reply = experiment_generator_agent.generate_reply(messages=messages)
            try:
                experiments = json.loads(reply).get("experiments", [])
            except (json.JSONDecodeError, TypeError):
                experiments = []
            self.untried_experiments = experiments.copy()
            if self.untried_experiments:
                idx = random.randrange(len(self.untried_experiments))
                exp = self.untried_experiments.pop(idx)
                self.tried_experiments.append(exp)

        if exp is not None:
            try:
                new_query = f"""\
Experiment plan: {exp.get('title', 'N/A')}

Objective: {exp.get('objective', 'N/A')}

Steps for the programmer:
{exp['steps']}

Deliverables:
{exp['deliverables']}"""
            except:
                # Recurse with get_next_experiment
                return self.get_next_experiment(experiment_generator_agent=experiment_generator_agent)

        return exp, new_query

    def has_untried_experiments(self):
        return bool(self.untried_experiments) or self.allow_generate_experiments

    def to_dict(self):
        return {
            "level": self.level,
            "node_idx": self.node_idx,
            "parent_idx": self.parent_idx,
            "query": self.query,
            "visits": self.visits,
            "value": self.value,
            "surprising": self.surprising,
            "belief_change": self.belief_change,
            "hypothesis": self.hypothesis,
            "prior": self.prior.to_dict() if self.prior else None,
            "posterior": self.posterior.to_dict() if self.posterior else None,
            "messages": self.messages,
            "creation_index": self.creation_index,
            "timestamp": self.timestamp,
        }

    def get_min_logs(self) -> list:
        """Returns minimal set of messages containing latest experiment, program, and analysis."""
        if not self.messages:
            return []

        latest_experiment = None
        latest_programmer = None
        latest_analyst = None

        for msg in reversed(self.messages):
            if not latest_experiment and msg.get("name") in ("user_proxy", "experiment_reviser"):
                latest_experiment = msg
            elif not latest_programmer and msg.get("name") == "experiment_programmer":
                latest_programmer = msg
            elif not latest_analyst and msg.get("name") == "experiment_analyst":
                latest_analyst = msg
            if latest_experiment and latest_programmer and latest_analyst:
                break

        return [m for m in [latest_experiment, latest_programmer, latest_analyst] if m]

    def get_parent_min_logs(self, k: Optional[int] = None) -> list:
        """Returns minimal set of messages from this node and all ancestor nodes.

        Args:
            k: Optional maximum number of parent levels to include. If None, includes all parents.
        """
        logs = self.get_min_logs()
        k_remaining = None if k is None else k - 1
        if self.parent and (k_remaining is None or k_remaining > 0):
            logs = self.parent.get_parent_min_logs(k=k_remaining) + logs
        return logs


def ucb1(node, exploration_weight=1.0):
    """Upper Confidence Bound 1 calculation for node selection"""
    if node.visits == 0:
        return float('inf')
    return (node.value / node.visits) + exploration_weight * np.sqrt(2 * np.log(node.parent.visits) / node.visits)


def run_mcts(
        query,
        dataset_paths,
        log_dirname,
        thread_id,
        work_dir,
        model_name="o4-mini",
        belief_model_name="gpt-4o",
        max_iterations=100,
        branching_factor=3,
        max_rounds=100000,
        selection_method=default_mcts_selection(1.0),
        allow_generate_experiments=False,
        n_belief_samples=30,
        use_min_context=False,
        root=None,
        nodes_by_level=None,
        k_logs=None,
        temperature=None,
        belief_temperature=None,
        reasoning_effort=None,
        implicit_bayes_posterior=False,
        surprisal_width=0.2,
        user_query=None,
        belief_mode="boolean",
        use_binary_reward=True
):
    """
    Run Monte Carlo Tree Search exploration using structured output agents.

    Args:
        dataset: The dataset to analyze
        query: The initial query to seed the exploration
        log_dirname: Where to save the logs
        thread_id: Unique identifier for this run
        max_iterations: Number of MCTS iterations to run
        max_depth: Maximum depth of the exploration tree
        branching_factor: Number of branches at each node
        exploration_weight: UCB1 exploration parameter
        temperature: LLM temperature parameter
        max_rounds: Maximum rounds of conversation
        allow_generate_experiments: Allow nodes (except root) to generate new experiments on demand
        n_belief_samples: Number of samples for belief distribution evaluation
        use_min_context: Use minimal context from parent nodes
        root: Root node to continue exploration from
        nodes_by_level: Nodes by level to continue exploration from
        k_logs: Number of parent levels to include in logs (None for all)
    """
    # Setup logger
    logger = TreeLogger(log_dirname)
    # Create work directory if it doesn't exist
    os.makedirs(work_dir, exist_ok=True)
    # Copy the dataset file paths to the working directory (to avoid modifying the original dataset)
    for dataset_fpath in dataset_paths:
        shutil.copy(dataset_fpath, work_dir)
    # Get the structured agents
    structured_agents = get_agents(work_dir, n_responses=1, model_name=model_name, temperature=temperature,
                                   reasoning_effort=reasoning_effort, branching_factor=branching_factor,
                                   user_query=user_query)
    agent_objs = {agent.name: agent for agent in structured_agents}
    user_proxy = agent_objs["user_proxy"]
    # Set up the group chat
    groupchat = setup_group_chat(agent_objs, max_rounds, thread_id)
    chat_manager = GroupChatManager(groupchat=groupchat, llm_config=None)
    first_iter_is_root = False
    # Initialize the root node if not continuing from saved state
    if root is None:
        root = MCTSNode(0, 0, None, None, untried_experiments=[query], allow_generate_experiments=False)
        nodes_by_level = defaultdict(list)
        nodes_by_level[0].append(root)
        first_iter_is_root = True

    experiment_generator_agent = agent_objs["experiment_generator"]

    try:
        for iteration_idx in range(max_iterations):
            # MCTS SELECTION, EXPANSION, and EXECUTION
            node = selection_method(root)

            exp, new_query = node.get_next_experiment(experiment_generator_agent=experiment_generator_agent)

            if new_query is not None:
                try:
                    new_level = node.level + 1
                    new_node_idx = len(nodes_by_level[new_level])
                    allow_generate = allow_generate_experiments if new_level > 0 else False
                    child = MCTSNode(new_level, new_node_idx, node.node_idx, new_query, parent=node,
                                     allow_generate_experiments=allow_generate)
                    node.children.append(child)
                    nodes_by_level[new_level].append(child)
                    node = child

                    # Update logger state
                    logger.level = node.level
                    logger.node_idx = node.node_idx

                    # Load parent message state
                    if node.parent_idx is not None and node.level - 1 > 0 and node.parent is not None:
                        if use_min_context:
                            parent_state = node.parent.get_parent_min_logs(k=k_logs)
                            # chat_manager.resume expects last manager to be starting message of chat so append it here
                            # The starting message is used in initiate_chat instead, as prior versions of autodv did (see exploration/agent_shell.py)
                            # (there might be overall more graceful ways to do this, but this is based on extending the existing code)
                            parent_state.append({"name": "user_proxy", "role": "user", "content": node.query})
                        else:
                            parent_state = logger.load_node(node.level - 1, node.parent_idx)
                        last_agent, last_message = chat_manager.resume(messages=parent_state)

                    # Generate new experiments
                    user_proxy.initiate_chat(recipient=chat_manager, message=node.query, clear_history=False)
                    logger.log_node(node.level, node.node_idx, chat_manager.messages_to_string(groupchat.messages))
                    node.messages = get_current_node_messages(groupchat.messages)

                    # Store generated experiments
                    try:
                        experiments = json.loads(groupchat.messages[-1]['content'])['experiments']
                    except (json.JSONDecodeError, KeyError, IndexError):
                        # Default to empty list if parsing fails or no experiments found
                        experiments = []
                    node.untried_experiments = experiments.copy()

                    # Calculate beliefs and rewards
                    reward = 0
                    if node.successful:
                        # Belief elicitation
                        if not first_iter_is_root or iteration_idx > 0:
                            is_surprisal, belief_change, prior, posterior = calculate_prior_and_posterior_beliefs(node,
                                                                                                                  model=belief_model_name,
                                                                                                                  temperature=belief_temperature,
                                                                                                                  reasoning_effort=reasoning_effort,
                                                                                                                  n_samples=n_belief_samples,
                                                                                                                  implicit_bayes_posterior=implicit_bayes_posterior,
                                                                                                                  surprisal_width=surprisal_width,
                                                                                                                  belief_mode=belief_mode)
                            node.surprising = is_surprisal
                            node.prior = prior
                            node.posterior = posterior
                            node.belief_change = belief_change

                            # Reward based on surprising hypothesis
                            reward = (1 if node.surprising else 0) if use_binary_reward else node.belief_change

                            # Print debug information
                            print(f"""\n\n\
================================================================================

NODE_LEVEL={node.level}, NODE_IDX={node.node_idx}:
-------------------------

Hypothesis: {node.hypothesis['hypothesis']}

Prior: {node.prior}
Posterior: {node.posterior}
Surprisal: {node.surprising}
Belief Change: {node.belief_change}
Reward: {reward}

================================================================================\n\n""")
                        else:
                            # For the root node, we don't calculate beliefs or rewards
                            node.surprising = None
                            node.prior = None
                            node.posterior = None
                            node.belief_change = None
                except OpenAIBadRequestError:
                    print(f"BadRequestError on iteration {iteration_idx} for node {node.level}_{node.node_idx}. "
                          "Skipping this node and continuing.")
                    continue

            # MCTS BACKPROPAGATION
            while node:
                node.visits += 1
                node.value += reward
                node = node.parent
            # Save the individual node after backpropagation
            node_file = os.path.join(log_dirname, f"mcts_node_{child.level}_{child.node_idx}.json")
            with open(node_file, "w") as f:
                json.dump(child.to_dict(), f, indent=2)
    except KeyboardInterrupt:
        print("\n\n######### EXPLORATION INTERRUPTED! SAVING THE CURRENT STATE... #########\n\n")

    # Save nodes to JSON
    nodes_list = save_nodes_to_json(nodes_by_level, log_dirname)

    # Save nodes to CSV
    csv_output_file = os.path.join(log_dirname, "mcts_nodes.csv")
    nodes_to_csv(nodes_list, csv_output_file)


def save_nodes_to_json(nodes_by_level, log_dirname):
    """Save all MCTS nodes to a JSON file.

    Args:
        nodes_by_level: Dictionary mapping levels to lists of MCTSNodes
        log_dirname: Directory to save the JSON file
    """
    node_data = []
    for level, nodes in nodes_by_level.items():
        for node in nodes:
            node_data.append(node.to_dict())
    output_file = os.path.join(log_dirname, "mcts_nodes.json")
    with open(output_file, "w") as f:
        json.dump(node_data, f, indent=2)
    return node_data


def load_mcts_from_json(json_obj_or_file, args):
    """Load and reconstruct MCTS nodes from a JSON log file.

    Args:
        json_obj_or_file: Path to the mcts_nodes.json file or the loaded JSON object.

    Returns:
        root: The root MCTSNode
        nodes_by_level: Dictionary mapping levels to lists of MCTSNodes
    """
    if type(json_obj_or_file) is str:
        with open(json_obj_or_file, 'r') as f:
            node_data = json.load(f)
    else:
        node_data = json_obj_or_file

    # Initialize storage
    nodes_by_level = defaultdict(list)
    node_map = {}  # Map (level, idx) to node objects for linking

    # First pass - create all nodes
    for data in node_data:
        level = data['level']
        node_idx = data['node_idx']
        parent_idx = data['parent_idx']

        # Create node (parent links added in second pass)
        node = MCTSNode(level, node_idx, parent_idx, data['query'])
        node.visits = data['visits']
        node.value = data['value']
        node.surprising = data['surprising']
        node.belief_change = data['belief_change']
        node.messages = data['messages']
        node.creation_index = data['creation_index']
        node.timestamp = data.get('timestamp', None)

        # All nodes except root can generate experiments
        if level > 0:
            node.allow_generate_experiments = True

        if data['prior']:
            node.prior = BELIEF_MODE_TO_CLS[args.belief_mode].DistributionFormat(**data['prior'])
        if data['posterior']:
            node.posterior = BELIEF_MODE_TO_CLS[args.belief_mode].DistributionFormat(**data['posterior'])

        nodes_by_level[level].append(node)
        node_map[(level, node_idx)] = node

    # Second pass - link parents and children
    for data in node_data:
        level = data['level']
        node_idx = data['node_idx']
        parent_idx = data['parent_idx']

        node = node_map[(level, node_idx)]
        if parent_idx is not None:
            parent = node_map[(level - 1, parent_idx)]
            node.parent = parent
            parent.children.append(node)

    root = nodes_by_level[0][0] if nodes_by_level[0] else None
    return root, nodes_by_level


def get_current_node_messages(messages):
    # Find last user_proxy message by iterating in reverse
    start_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("name") == "user_proxy":
            start_idx = i
            break

    if start_idx is None:
        return None, []

    node_messages = messages[start_idx:]

    return node_messages


def setup_group_chat(agents, max_rounds, thread_id):
    """Set up the group chat with agents and rules"""
    groupchat = autogen.GroupChat(
        agents=list(agents.values()),
        messages=[],
        max_round=max_rounds,
        speaker_selection_method=SpeakerSelector().select_next_speaker
    )
    return groupchat


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
    dataset_paths = get_datasets_fpaths(args.dataset_metadata)

    if args.continue_from_dir or args.continue_from_json:
        if args.continue_from_dir:
            # Load nodes from directory
            resumption_obj = collect_logs_for_resumption(args.continue_from_dir)
            root, nodes_by_level = load_mcts_from_json(resumption_obj, args)
            # Copy all files except args.json from continue_from_dir to the new log directory
            for filename in os.listdir(args.continue_from_dir):
                if filename != "args.json":
                    shutil.copy(os.path.join(args.continue_from_dir, filename), os.path.join(log_dirname, filename))
        else:
            # Load from collected JSON file
            root, nodes_by_level = load_mcts_from_json(args.continue_from_json, args)
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
        load_dataset_title = "[ROOT] Data Loading"
        load_dataset_objective = "Load the dataset and generate summary statistics. "
        load_dataset_steps = f"1. Load the dataset(s) at {[os.path.basename(dp) for dp in dataset_paths]}.\n2. Generate summary statistics for the dataset(s)."
        load_dataset_deliverables = "1. Dataset(s) loaded.\n2. Summary statistics generated."

        if args.dataset_metadata_type == 'blade':
            load_dataset_objective += f"Here is the dataset metadata:\n\n{get_blade_description(args.dataset_metadata)}"
        else:  # DiscoveryBench-style
            load_dataset_objective += f"Here is the dataset metadata:\n\n{get_dataset_description(args.dataset_metadata)}"

        query = {
            "title": load_dataset_title,
            "objective": load_dataset_objective,
            "steps": load_dataset_steps,
            "deliverables": load_dataset_deliverables
        }

    # Set up selection method based on args
    if args.use_beam_search:
        selection_method = beam_search(args.k_experiments, args.beam_width, args.out_dir)
        selection_method_name = "beam_search"
    elif (args.pw_k is not None) and (args.pw_alpha is not None):
        selection_method = progressive_widening(args.pw_k, args.pw_alpha, args.exploration_weight)
        selection_method_name = "progressive_widening"
    else:
        selection_method = default_mcts_selection(args.exploration_weight)
        selection_method_name = "ucb1"
    print(f"Tree search selection method: {selection_method}\n")

    # Run exploration
    run_mcts(
        query=query,
        dataset_paths=dataset_paths,
        log_dirname=log_dirname,
        thread_id=timestamp,
        work_dir=work_dirname,
        max_iterations=remaining_iters,
        branching_factor=args.k_experiments,
        selection_method=selection_method,
        allow_generate_experiments=args.allow_generate_experiments,
        n_belief_samples=args.n_belief_samples,
        use_min_context=args.use_min_context,
        root=root,
        nodes_by_level=nodes_by_level,
        k_logs=args.k_logs,
        model_name=args.model,
        belief_model_name=args.belief_model,
        temperature=args.temperature,
        belief_temperature=args.belief_temperature,
        reasoning_effort=args.reasoning_effort,
        implicit_bayes_posterior=args.implicit_bayes_posterior,
        surprisal_width=args.surprisal_width,
        user_query=args.user_query,
        belief_mode=args.belief_mode,
        use_binary_reward=args.use_binary_reward
    )

    if args.delete_work_dir:
        shutil.rmtree(args.work_dir)
        print(f"\nDELETED WORKING DIRECTORY: {args.work_dir}")

    print(f"\nRUN FINISHED!\n\nLOGS: {log_dirname}")

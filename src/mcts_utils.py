import json
import os
from collections import defaultdict
from typing import List, Dict

from autogen import GroupChat, GroupChatManager

from src.beliefs import BELIEF_MODE_TO_CLS
from src.deduplication import dedupe
from src.mcts import MCTSNode
from src.nodes_to_csv import nodes_to_csv
from src.reconstruct_mcts import collect_logs_for_resumption
from src.transitions import SpeakerSelector


def load_mcts_from_json(json_obj_or_file_or_dir, args):
    """Load and reconstruct MCTS nodes from a JSON object, log file, or directory.

    Args:
        json_obj_or_file_or_dir: Loaded JSON object or a path to the mcts_nodes.json file or to a directory with mcts_node_*.json files.

    Returns:
        root: The root MCTSNode
        nodes_by_level: Dictionary mapping levels to lists of MCTSNodes
    """
    if type(json_obj_or_file_or_dir) is str:
        if os.path.isdir(json_obj_or_file_or_dir):
            node_data = collect_logs_for_resumption(json_obj_or_file_or_dir)
        else:
            with open(json_obj_or_file_or_dir, 'r') as f:
                node_data = json.load(f)
    else:
        node_data = json_obj_or_file_or_dir

    # Initialize storage
    nodes_by_level = defaultdict(list)
    node_map = {}  # Map (level, idx) to node objects for linking

    # First pass - create all nodes
    for data in node_data:
        level = data['level']
        node_idx = data['node_idx']
        parent_idx = data['parent_idx']

        # Create node (parent links added in second pass)
        node = MCTSNode(level=level, node_idx=node_idx, parent_idx=parent_idx, hypothesis=data['hypothesis'],
                        query=data['query'])
        node.visits = data['visits']
        node.value = data['value']
        node.surprising = data['surprising']
        node.belief_change = data.get('belief_change', None)
        node.messages = data['messages']
        node.creation_index = data['creation_index']
        node.timestamp = data.get('timestamp', None)

        # All nodes except root can generate experiments
        if level > 0:
            node.allow_generate_experiments = args.allow_generate_experiments

        if data['prior']:
            node.prior = BELIEF_MODE_TO_CLS[data['prior']['_type']].DistributionFormat(**data['prior'])
        if data['posterior']:
            node.posterior = BELIEF_MODE_TO_CLS[data['posterior']['_type']].DistributionFormat(**data['posterior'])

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


def save_nodes(nodes_dict_or_list, log_dirname, run_dedupe=True, model="gpt-4o", save_csv=True):
    if type(nodes_dict_or_list) in [dict, defaultdict]:
        nodes_list = []
        for level, nodes in nodes_dict_or_list.items():
            if level == 0:
                continue
            for node in nodes:
                nodes_list.append(node.to_dict())
    else:
        nodes_list = nodes_dict_or_list

    # Save nodes to JSON
    nodes_list = save_nodes_to_json(nodes_list, log_dirname, run_dedupe=run_dedupe, dedupe_model=model)
    # Save nodes to CSV
    if save_csv:
        csv_output_file = os.path.join(log_dirname, "mcts_nodes.csv")
        nodes_to_csv(nodes_list, csv_output_file)


def save_nodes_to_json(nodes_list, log_dirname, run_dedupe=True, dedupe_model="gpt-4o", log_dedupe_comparisons=False):
    """Save all MCTS nodes to a JSON file.

    Args:
        nodes_list: List of MCTS node objects.
        log_dirname: Directory to save the JSON file
    """
    # Optionally deduplicate nodes based on hypothesis
    if run_dedupe:
        deduped_nodes, _, _ = dedupe(nodes_list, model=dedupe_model,
                                     log_comparisons_fname=None if not log_dedupe_comparisons else os.path.join(
                                         log_dirname, "dedupe_comparisons.json"))
        file_to_save = deduped_nodes
    else:
        file_to_save = nodes_list

    output_file = os.path.join(log_dirname, "mcts_nodes.json")
    with open(output_file, "w") as f:
        json.dump(file_to_save, f, indent=2)
    print(f"[JSON] MCTS nodes (n={len(file_to_save)}) saved to {output_file}.\n")
    # Also save the original nodes list for reference
    original_nodes_file = os.path.join(log_dirname, "mcts_nodes_all.json")
    with open(original_nodes_file, "w") as f:
        json.dump(nodes_list, f, indent=2)
    print(f"[JSON] Original MCTS nodes (n={len(nodes_list)}) saved to {original_nodes_file}.\n")
    return file_to_save


def get_msgs_from_latest_query(messages):
    # Find last user_proxy message by iterating in reverse
    start_idx = None
    for i, message in enumerate(reversed(messages)):
        if message.get("name") == "user_proxy":
            start_idx = len(messages) - 1 - i
            break
    if start_idx is None:
        return []
    node_messages = messages[start_idx:]
    return node_messages


def setup_group_chat(agents, max_rounds):
    # Set up the group chat with agents and rules
    group_chat = GroupChat(
        agents=list(agents.values()),
        messages=[],
        max_round=max_rounds,
        speaker_selection_method=SpeakerSelector().select_next_speaker
    )
    chat_manager = GroupChatManager(groupchat=group_chat, llm_config=None)
    return group_chat, chat_manager


def get_nodes(in_fpath_or_json: str | List[Dict[str, any]]) -> List[Dict[str, any]] | None:
    """
    Load MCTS nodes from a file, directory, or a list of dictionaries without creating class objects.
    Args:
        in_fpath_or_json: Path to the MCTS nodes JSON file, a directory containing MCTS node files, or a list of MCTS nodes as dictionaries.

    Returns:
        List of MCTS nodes as dictionaries.
    """
    if type(in_fpath_or_json) is list:
        mcts_nodes = in_fpath_or_json
    else:
        # Load the MCTS nodes from the input file
        if os.path.isdir(in_fpath_or_json):
            mcts_nodes = []
            for filename in os.listdir(in_fpath_or_json):
                if filename.startswith('mcts_node_') and filename.endswith('.json'):
                    with open(os.path.join(in_fpath_or_json, filename), 'r') as f:
                        obj = json.load(f)
                        mcts_nodes.append(obj)
        else:
            with open(in_fpath_or_json, 'r') as f:
                mcts_nodes = json.load(f)
    return mcts_nodes


def print_node_info(node):
    print(f"""\n\n\
================================================================================

NODE_LEVEL={node.level}, NODE_IDX={node.node_idx}:
-------------------------

Hypothesis: {node.hypothesis}
Prior: {node.prior.get_mean_belief()}
Posterior: {node.posterior.get_mean_belief(prior=node.prior)}
Belief Change: {node.belief_change}
Surprisal: {node.surprising}

================================================================================\n\n""")

import copy
import os
from typing import Optional
import random
import json
from datetime import datetime

import numpy as np

from src.mcts_utils import get_query_from_experiment, get_experiment_from_query, get_node_level_idx, get_context_string
from src.utils import try_loading_dict


class MCTSNode(object):
    _creation_counter = 0

    def __init__(self, level=None, node_idx=None, hypothesis=None, query=None, parent_idx=None, parent=None,
                 untried_experiments=None, allow_generate_experiments=False, experiment_plan=None, code=None,
                 code_output=None, analysis=None, review=None, creation_idx=None):
        # Tree attributes
        self.creation_idx = creation_idx if creation_idx is not None else MCTSNode._creation_counter
        MCTSNode._creation_counter += 1  # Used to replay MCTS from log files
        self.time_elapsed = None
        self.level = level
        self.node_idx = node_idx
        self.id = f"node_{self.level}_{self.node_idx}" if self.level is not None and self.node_idx is not None else None
        self.children = []
        self.parent = parent  # MCTSNode
        if self.parent is not None:
            self.parent_idx = self.parent.node_idx
            self.parent_id = self.parent.id
            self.parent.children.append(self)
        else:
            self.parent_idx = parent_idx
            self.parent_id = f"node_{self.level - 1}_{self.parent_idx}" if self.parent_idx is not None else None
        self.success = None

        # Agent attributes
        self.query = query
        self.messages = []
        self.untried_experiments = copy.deepcopy(untried_experiments) if untried_experiments is not None else []
        self.tried_experiments = []  # Track all tried experiments
        self.allow_generate_experiments = allow_generate_experiments

        # Experiment attributes
        self.hypothesis = hypothesis
        self.experiment_plan = experiment_plan
        self.code = code
        self.code_output = code_output
        self.analysis = analysis
        self.review = review

        # MCTS attributes
        self.visits = 0  # Visits to this node or its children
        self.value = 0.  # Number of surprising hypotheses
        self.self_value = 0.  # Value of this node only (not aggregated from children)

        # Belief attributes
        self.surprising: Optional[bool] = None
        self.prior = None
        self.posterior = None
        self.belief_change: Optional[float] = None  # Change in belief from prior to posterior
        self.kl_divergence: Optional[float] = None

    def init_from_dict(self, data):
        """Initialize node attributes from a dictionary."""
        # Tree attributes
        self.creation_idx = data.get('creation_idx', MCTSNode._creation_counter)
        self.time_elapsed = data.get('time_elapsed', self.time_elapsed)
        self.id = data.get('id', None)
        if self.id is not None:
            self.level, self.node_idx = get_node_level_idx(self.id)
        else:
            self.level = data['level']
            self.node_idx = data['node_idx']
            self.id = f"node_{self.level}_{self.node_idx}"
        self.parent_id = data.get('parent_id', self.parent_id)
        if self.parent_id is not None:
            _, self.parent_idx = get_node_level_idx(self.parent_id)
        else:
            self.parent_idx = data.get('parent_idx', self.parent_idx)
            if self.parent_idx is not None:
                self.parent_id = f"node_{self.level - 1}_{self.parent_idx}"
        self.success = data.get('success', self.success)

        # Agent attributes
        self.query = data.get('query', "N/A")
        self.messages = data.get('messages', self.messages)
        self.untried_experiments = data.get('untried_experiments', self.untried_experiments)
        # self.tried_experiments = data.get('tried_experiments', self.tried_experiments)
        self.allow_generate_experiments = self.allow_generate_experiments and self.level > 0

        # Experiment attributes
        self.hypothesis = data.get('hypothesis', self.hypothesis)
        self.experiment_plan = data.get('experiment_plan', self.experiment_plan)
        self.code = data.get('code', self.code)
        self.code_output = data.get('code_output', self.code_output)
        self.analysis = data.get('analysis', self.analysis)
        self.review = data.get('review', self.review)

        # MCTS attributes
        self.visits = data.get('visits', self.visits)
        self.value = data.get('value', self.value)
        self.self_value = data.get('self_value', self.self_value)

        # Belief attributes
        self.surprising = data.get('surprising', self.surprising)
        from src.beliefs import BELIEF_MODE_TO_CLS  # Import here to avoid circular import issues
        if 'prior' in data and data['prior']:
            belief_cls = BELIEF_MODE_TO_CLS[data['prior']['_type']]
            self.prior = belief_cls.DistributionFormat(**data['prior'])
        if 'posterior' in data and data['posterior']:
            belief_cls = BELIEF_MODE_TO_CLS[data['posterior']['_type']]
            self.posterior = belief_cls.DistributionFormat(**data['posterior'])
        self.belief_change = data.get('belief_change', self.belief_change)
        self.kl_divergence = data.get('kl_divergence', self.kl_divergence)

    def get_next_experiment(self, experiment_generator=None, n_retry=3):
        """
        Returns the next untried experiment. If none left and generating experiments is allowed, generates more using
        the experiment generator agent.
        """
        new_experiment, new_query = None, None

        if n_retry > 0:
            if self.untried_experiments:
                idx = random.randrange(len(self.untried_experiments))
                new_experiment = self.untried_experiments.pop(idx)
                self.tried_experiments.append(new_experiment)
            elif self.allow_generate_experiments and experiment_generator is not None:
                # Generate new experiments on-demand, providing all previous experiments as context
                _messages = self.messages + [{
                    "role": "user",
                    "content": f"Generate new experiments given these previously attempted experiments: {json.dumps(self.tried_experiments)}"
                }]
                _reply = experiment_generator.generate_reply(messages=_messages)
                try:
                    experiments = try_loading_dict(_reply).get("experiments", [])
                except (json.JSONDecodeError, TypeError):
                    experiments = []
                self.untried_experiments = experiments.copy()
                if self.untried_experiments:
                    idx = random.randrange(len(self.untried_experiments))
                    new_experiment = self.untried_experiments.pop(idx)
                    self.tried_experiments.append(new_experiment)

            if new_experiment is not None:
                try:
                    new_query = get_query_from_experiment(new_experiment)
                except:
                    pass
            if new_query is None:
                return self.get_next_experiment(experiment_generator=experiment_generator, n_retry=n_retry - 1)

        return new_experiment, new_query

    def has_untried_experiments(self):
        return bool(self.untried_experiments) or self.allow_generate_experiments

    def to_dict(self):
        return {
            "id": self.id,
            "success": self.success,
            "parent_id": self.parent_id,
            "creation_idx": self.creation_idx,
            "time_elapsed": self.time_elapsed,
            "visits": self.visits,
            "value": self.value,
            "self_value": self.self_value,
            "surprising": self.surprising,
            "belief_change": self.belief_change,
            "kl_divergence": self.kl_divergence,
            "prior": self.prior.to_dict() if self.prior else None,
            "posterior": self.posterior.to_dict() if self.posterior else None,
            "hypothesis": self.hypothesis,
            "experiment_plan": self.experiment_plan,
            "code": self.code,
            "code_output": self.code_output,
            "analysis": self.analysis,
            "review": self.review,
            "untried_experiments": self.untried_experiments,
            "tried_experiments": self.tried_experiments,
            "query": self.query,
            "messages": self.messages,
        }

    def read_experiment_from_messages(self, store_new_experiments=False):
        """Extracts experiment details from messages and updates the node's attributes."""
        latest_experiment = None
        was_revised = False
        latest_programmer = None
        latest_code_executor = None
        latest_analyst = None
        latest_reviewer = None
        latest_reviewer_feedback = "N/A"
        latest_reviewer_success = False
        latest_experiment_generator = None

        for msg in reversed(self.messages):
            if not latest_experiment and msg.get("name") in ["user_proxy", "experiment_reviser"]:
                latest_experiment = msg.get("content")
                if msg.get("name") == "experiment_reviser":
                    was_revised = True
            elif not latest_programmer and msg.get("name") == "experiment_programmer":
                latest_programmer = try_loading_dict(msg.get("content")).get("code", "N/A")
            elif not latest_code_executor and msg.get("name") == "code_executor":
                latest_code_executor = msg.get("content")
            elif not latest_analyst and msg.get("name") in ["experiment_analyst", "experiment_code_analyst"]:
                latest_analyst = try_loading_dict(msg.get("content")).get("analysis", "N/A")
            elif not latest_reviewer and msg.get("name") == "experiment_reviewer":
                latest_reviewer = try_loading_dict(msg.get("content"))
                latest_reviewer_feedback = latest_reviewer.get("feedback", "N/A")
                if latest_reviewer_feedback == "":
                    latest_reviewer_feedback = "N/A"
                latest_reviewer_success = latest_reviewer.get("success", False)
            elif not latest_experiment_generator and msg.get("name") == "experiment_generator":
                latest_experiment_generator = try_loading_dict(msg.get("content")).get("experiments", [])

            if (latest_experiment and latest_programmer and
                    latest_code_executor and latest_analyst and latest_reviewer):
                break

        if was_revised:
            latest_experiment_obj = try_loading_dict(latest_experiment)
            # Change what the query should now be based on the revised experiment
            self.query = get_query_from_experiment(latest_experiment_obj)
        else:
            latest_experiment_obj = get_experiment_from_query(latest_experiment)  # assuming it is a query string

        self.hypothesis = latest_experiment_obj.get("hypothesis", "N/A")
        self.experiment_plan = latest_experiment_obj.get("experiment_plan", "N/A")
        self.code = latest_programmer
        self.code_output = latest_code_executor
        self.analysis = latest_analyst
        self.review = latest_reviewer_feedback
        self.success = latest_reviewer_success

        # Store new experiments into untried_experiments
        if store_new_experiments and latest_experiment_generator:
            self.untried_experiments += latest_experiment_generator

    def get_context(self, include_code_output=False) -> None | str:
        """Returns the node's hypothesis, experiment, output, analysis, and review."""
        if len(self.messages) == 0:
            return None
        context_str = get_context_string(self.query, self.code_output, self.analysis, self.review,
                                         include_code_output=include_code_output)
        return context_str

    def get_path_context(self, k: Optional[int] = None, skip_root=True) -> None | list:
        """Returns messages from the node to the root

        Args:
            k: Optional maximum number of parent levels to include. If None, includes all parents.
            skip_root: If True, skips the root node in the context.
        """
        context = self.parent.get_context() if self.parent is not None else None
        if context is not None:
            if skip_root and self.parent.level <= 1:
                return []
            context = [context]
        k_remaining = None if k is None else k - 1
        if context is not None and self.parent is not None and (k_remaining is None or k_remaining > 0):
            parent_context = self.parent.get_path_context(k=k_remaining, skip_root=skip_root)
            if parent_context is not None:
                context = parent_context + context
        return context

    def update_counts(self, visits: int = 1, reward: float = 0):
        """Update the visit count and value of this node and its parents."""
        self.visits += visits
        self.value += reward
        if self.parent is not None:
            self.parent.update_counts(visits=visits, reward=reward)


def default_mcts_selection(exploration_weight):
    def select(node, nodes_by_level):
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
        exploration_weight: Exploration weight for UCB1 selection method.

    Returns:
        A callable function that accepts a `node` and returns the selected node.
    """

    def select(node, nodes_by_level):
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
            return select(max(node.children, key=lambda n: ucb1(n, exploration_weight)), nodes_by_level)

        # If no children exist, return the current node
        return node

    return select


def progressive_widening_all(k, alpha, exploration_weight=1.0):
    """
    Create a progressive widening selection function with an alternative implementation that selects the node
    which has the highest UCB1 value from the entire set of nodes.

    Args:
        k: Progressive widening constant.
        alpha: Progressive widening exponent.
        exploration_weight: Exploration weight for UCB1 selection method.

    Returns:
        A callable function that accepts a `node` and returns the selected node.
    """

    def select(node, nodes_by_level):
        all_nodes = [n for level, nodes in nodes_by_level.items() if level > 0 for n in nodes]
        # Sort all nodes by UCB1 value
        all_nodes_sorted = sorted(all_nodes, key=lambda n: ucb1(n, exploration_weight), reverse=True)

        for _node in all_nodes_sorted:
            num_visits = _node.visits
            num_children = len(_node.children)
            # If we can add a new child based on the progressive widening condition
            if (num_children < k * (num_visits ** alpha)) and _node.has_untried_experiments():
                # Sample a new child (expand the tree)
                return _node

        # If no children exist, return the current node
        return node

    return select


def ucb1_recursive(exploration_weight=1.0):
    """
    Create a UCB1 traversal selection function.

    Args:
        exploration_weight: Exploration weight for UCB1 selection method.

    Returns:
        A callable function that accepts a `node` and returns the selected node.
    """

    def select(node, nodes_by_level):
        # Sort self and children by UCB1 value
        all_nodes = [node] + node.children
        sorted_nodes = sorted(all_nodes, key=lambda n: ucb1(n, exploration_weight), reverse=True)

        for best_node in sorted_nodes:
            if best_node.has_untried_experiments():
                if best_node is node:
                    return best_node
                return select(best_node, nodes_by_level)

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

    def select(root, nodes_by_level):
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
            return select(root, nodes_by_level)  # Recurse with new beam

        return beam[0]  # Default to first beam node if no children

    return select


def ucb1(node, exploration_weight=1.0):
    """Upper Confidence Bound 1 calculation for node selection"""
    if node.visits == 0:
        return float('inf')
    exploitation_term = node.value / node.visits
    exploration_term = 0
    if node.parent:
        exploration_term = np.sqrt(2 * np.log(node.parent.visits) / node.visits)
    return exploitation_term + exploration_weight * exploration_term

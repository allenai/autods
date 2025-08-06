import os
from typing import Optional
import random
import json
from datetime import datetime

import numpy as np

from src.utils import try_loading_dict


class MCTSNode(object):
    _creation_counter = 0

    def __init__(self, level, node_idx, hypothesis, query, parent_idx=None, parent=None, untried_experiments=None,
                 allow_generate_experiments=False):
        self.level = level
        self.node_idx = node_idx
        self.id = f"node_{self.level}_{self.node_idx}"
        self.parent: Optional[MCTSNode] = parent
        if self.parent is not None:
            self.parent_idx = self.parent.node_idx
            self.parent_id = self.parent.id
        else:
            self.parent_idx = parent_idx
            self.parent_id = f"node_{self.level - 1}_{self.parent_idx}" if self.parent_idx is not None else None
        self.hypothesis = hypothesis
        self.query = query
        self.children = []
        self.visits = 0  # Visits to this node or its children
        self.value = 0.  # Number of surprising hypotheses
        self.self_value = 0.  # Value of this node only (not aggregated from children)
        self.untried_experiments = untried_experiments.copy() if untried_experiments is not None else []
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
        exp, hypothesis, new_query = None, None, None

        if self.untried_experiments:
            idx = random.randrange(len(self.untried_experiments))
            exp = self.untried_experiments.pop(idx)
            self.tried_experiments.append(exp)
        elif self.allow_generate_experiments and experiment_generator_agent is not None:
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
                hypothesis = exp['hypothesis']
                exp_plan = exp['experiment_plan']
                new_query = ""
                if hypothesis is not None:
                    new_query += f"Hypothesis: {hypothesis}\n\n"
                new_query += f"""\
Experiment objective: {exp_plan['objective']}

Steps for the programmer:
{exp_plan['steps']}

Deliverables:
{exp_plan['deliverables']}"""
            except Exception as e:
                print(f"Error processing proposed experiment: {e}")
                print("Getting next experiment...")
                # Recurse with get_next_experiment
                return self.get_next_experiment(experiment_generator_agent=experiment_generator_agent)

        return exp, hypothesis, new_query

    def has_untried_experiments(self):
        return bool(self.untried_experiments) or self.allow_generate_experiments

    def to_dict(self):
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "level": self.level,
            "node_idx": self.node_idx,
            "parent_idx": self.parent_idx,
            "creation_index": self.creation_index,
            "timestamp": self.timestamp,
            "hypothesis": self.hypothesis,
            "visits": self.visits,
            "value": self.value,
            "self_value": self.self_value,
            "surprising": self.surprising,
            "belief_change": self.belief_change,
            "prior": self.prior.to_dict() if self.prior else None,
            "posterior": self.posterior.to_dict() if self.posterior else None,
            "query": self.query,
            "messages": self.messages,
        }

    def get_context(self, include_code_output=False) -> None | str:
        """Returns minimal set of messages containing latest experiment, program, and analysis."""
        if len(self.messages) == 0:
            return None

        latest_experiment = None
        latest_programmer = None
        latest_code_executor = None
        latest_analyst = None
        latest_reviewer = None

        for msg in reversed(self.messages):
            if not latest_experiment and msg.get("name") in ["user_proxy", "experiment_reviser"]:
                latest_experiment = msg.get("content")
            elif not latest_programmer and msg.get("name") == "experiment_programmer":
                latest_programmer = try_loading_dict(msg.get("content")).get("code", "N/A")
            elif not latest_code_executor and msg.get("name") == "code_executor":
                latest_code_executor = msg.get("content")
            elif not latest_analyst and msg.get("name") in ["experiment_analyst", "experiment_code_analyst"]:
                latest_analyst = try_loading_dict(msg.get("content")).get("analysis", "N/A")
            elif not latest_reviewer and msg.get("name") == "experiment_reviewer":
                latest_reviewer = try_loading_dict(msg.get("content")).get("feedback", "N/A")
                if latest_reviewer == "":
                    latest_reviewer = "N/A"

            if latest_experiment and latest_programmer and latest_code_executor and latest_analyst and latest_reviewer:
                break

        context_str = latest_experiment + "\n\n"

        if include_code_output:
            context_str += f"Code Output:\n{latest_code_executor}\n\n"

        context_str += f"""\
Analysis: {latest_analyst}

Review: {latest_reviewer}"""

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
        exploration_weight: Exploration weight for UCB1 selection method.

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


def ucb1(node, exploration_weight=1.0):
    """Upper Confidence Bound 1 calculation for node selection"""
    if node.visits == 0:
        return float('inf')
    return (node.value / node.visits) + exploration_weight * np.sqrt(2 * np.log(node.parent.visits) / node.visits)

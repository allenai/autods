import os
from typing import List, Dict
from pydantic import ValidationError
from openai import OpenAI
import json


def query_llm(
        messages: List[Dict[str, str]],
        n_samples: int,
        model: str = "gpt-4o",
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        response_format=None,
        client: OpenAI = None
):
    if client is None:
        client = OpenAI()
    n_samples_batch_size = 8 if model.startswith("o") else n_samples
    responses = []
    # Sample exactly n_samples responses
    for i in range(0, n_samples, n_samples_batch_size):
        kwargs = {
            "model": model,
            "messages": messages,
            "n": min(n_samples_batch_size, n_samples - len(responses))
        }
        if not model.startswith("o") and temperature is not None:
            kwargs["temperature"] = temperature
        if model.startswith("o") and reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort

        if response_format is not None:
            kwargs["response_format"] = response_format

        try:
            response = client.chat.completions.parse(**kwargs)
        except ValidationError:
            # Retry if the response format validation fails
            response = client.chat.completions.parse(**kwargs)

        for choice in response.choices:
            if choice.message.content is None:
                continue
            responses += [json.loads(choice.message.content)]
    return responses


def get_nodes(in_fpath_or_json: str | List[Dict[str, any]]) -> List[Dict[str, any]] | None:
    """
    Load MCTS nodes from a file, directory, or a list of dictionaries.
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

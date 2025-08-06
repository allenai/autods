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


def try_loading_dict(_dict_str):
    try:
        return json.loads(_dict_str)
    except json.JSONDecodeError:
        try:
            return json.loads(_dict_str + '"}')  # Fix case where string is truncated
        except json.JSONDecodeError:
            return {}

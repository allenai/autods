import os
import json
from typing import List, Dict

import numpy as np
import boto3
from pydantic import ValidationError
from openai import OpenAI


def query_llm(
    messages: List[Dict[str, str]],
    n_samples: int,
    model: str = "gpt-4o",
    temperature: float | None = None,
    reasoning_effort: str | None = None,
    response_format=None,
    client: OpenAI = None,
):
    if client is None:
        client = OpenAI()
    is_reasoning_model = any(model.startswith(prefix) for prefix in ["o", "gpt-5"])

    n_samples_batch_size = 8 if is_reasoning_model else n_samples
    responses = []
    # Sample exactly n_samples responses
    for i in range(0, n_samples, n_samples_batch_size):
        kwargs = {
            "model": model,
            "messages": messages,
            "n": min(n_samples_batch_size, n_samples - len(responses)),
        }
        if not is_reasoning_model and temperature is not None:
            kwargs["temperature"] = temperature
        if is_reasoning_model and reasoning_effort is not None:
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


def fuse_gaussians(means, stds, weight=1.0):
    """
    Fuse n independent Gaussian beliefs N(mu_i, sigma_i^2)
    into a single Gaussian via product of Gaussians.

    Parameters
    ----------
    means : array-like, shape (n,)
        The means μ_i of the Gaussian beliefs.
    stds : array-like, shape (n,)
        The standard deviations σ_i of the Gaussian beliefs.
    weight : float, optional
        A weight to apply to the precision of each Gaussian. Default is 1.0.

    Returns
    -------
    mu_star : float
        The fused mean μ_*.
    sigma_star : float
        The fused standard deviation σ_*.
    """
    means = np.array(means, dtype=float)
    variances = (
        np.array(stds, dtype=float) ** 2 + 1e-10
    )  # Add small value to avoid division by zero

    # Precisions
    precisions = weight / variances

    # Combined precision and variance
    precision_star = np.sum(precisions)
    variance_star = 1.0 / precision_star

    # Combined mean
    mu_star = np.sum(precisions * means) / precision_star
    sigma_star = np.sqrt(variance_star)

    return mu_star, sigma_star


def fetch_from_s3(links: List[str], download_dir="_s3") -> List[str]:
    """
    Download data from S3 URLs
    Attributes:
        links (List[str]): List of S3 URLs to download
        download_dir (str): Directory to save downloaded files
    Returns:
        List of local file paths where files are downloaded
    """
    s3_client = boto3.client("s3")
    fpaths = []
    for link in links:
        _, _, bucket, key = link.split("/", 3)
        local_file_path = os.path.join(download_dir, key)
        local_dir = os.path.dirname(local_file_path)
        os.makedirs(local_dir, exist_ok=True)
        byte_str = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read()
        with open(local_file_path, "wb") as file:
            file.write(byte_str)
        fpaths.append(local_file_path)

    return fpaths

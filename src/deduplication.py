import argparse
import json
import random
import numpy as np
from openai import OpenAI
from scipy.cluster.hierarchy import linkage
from pydantic import BaseModel, Field
from utils import query_llm, get_nodes


class ArgParser(argparse.ArgumentParser):
    def __init__(self, group=None):
        super().__init__(description='Get surprising nodes from MCTS logs')
        self.add_argument('--in_fpath', type=str, required=True,
                          help='mcts_nodes.json file path or directory containing mcts_node_*.json files')
        self.add_argument('--out_fpath', type=str, help='output directory for clusters and labels')
        self.add_argument('--n_samples', type=int, default=30, help='Number of samples for LLM decisions')
        self.add_argument('--merge_threshold', type=float, default=0.7,
                          help='Threshold for merging hypotheses based on LLM decisions')
        self.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')


def hyp_dict_to_str(d):
    return (f"Hypothesis: {d.get('hypothesis', 'N/A')}\n"
            f"Contexts: {d.get('contexts', d.get('context', 'N/A'))}\n"
            f"Variables: {d.get('variables', 'N/A')}\n"
            f"Relationships: {d.get('relationships', 'N/A')}")


def get_structured_hypothesis(node):
    level, node_idx = node.get("level"), node.get("node_idx")
    h = node.get("hypothesis", None)
    if h is None:
        return None
    hyp_str = h.get("hypothesis", "")
    dims = h.get("dimensions", {
        "contexts": [],
        "variables": [],
        "relationships": []
    })
    return {
        "node_id": f"node_{level}_{node_idx}",
        "hypothesis": hyp_str,
        **dims
    }


def get_structured_hypotheses(nodes_or_json_path):
    raw_nodes_list = get_nodes(nodes_or_json_path)
    node_list = [hyp for node in raw_nodes_list if (hyp := get_structured_hypothesis(node)) is not None]
    return node_list


def get_embedding(texts, model="text-embedding-3-large", batch_size=128, client=None):
    """
    Compute embeddings for a list of texts using the OpenAI Embeddings API.
    Args:
        texts (list): A list of text strings to be embedded.
        model (str, optional): The identifier for the embedding model to use.
        batch_size (int, optional): The number of texts to process in one API call.
    Returns:
        numpy.ndarray: An array of embeddings for the input texts.
    """
    if client is None:
        client = OpenAI()
    all_embeddings = []
    # Process the texts in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        # Request embeddings for the current batch from the API
        response = client.embeddings.create(input=batch, model=model)
        for item in response.data:
            # Convert the embedding to a NumPy array and add it to the list
            all_embeddings.append(np.array(item.embedding))
    return np.array(all_embeddings)


def get_llm_merge_decision(hyp1: str, hyp2: str, n_samples: int = 30, threshold: float = 0.7, model: str = "gpt-4o",
                           temperature: float = 1.0, reasoning_effort: str = "medium"):
    class ResponseFormat(BaseModel):
        is_same: bool = Field(..., description="Whether the two hypotheses are the same or not.")

    system_prompt = "You are a research scientist skilled at analyzing statistical hypotheses."
    prompt = (
        f"You are given two hypothesis sets. Each set describes a single hypothesis structured into a context for the "
        f"hypothesis, the variables involved, and the statistical relationships between the variables under that "
        f"context. Your task is to determine whether both sets represent the same hypothesis or not.\n\n"
        f"Hypothesis Set 1:\n{hyp1}\n\nHypothesis Set 2:\n{hyp2}"
    )
    all_msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    response = query_llm(all_msgs, model=model, n_samples=n_samples,
                         temperature=temperature, reasoning_effort=reasoning_effort,
                         response_format=ResponseFormat)
    true_prop = sum([1 for _res in response if _res["is_same"]]) / n_samples

    return true_prop >= threshold


def dedupe(nodes_or_json_path, n_samples=30, merge_threshold=0.7, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    data = get_structured_hypotheses(nodes_or_json_path)

    dedup_hyp, dedup_struct_hyp, hyp_to_index, orig_to_dedup = [], [], {}, []

    # Deduplicate hypotheses by exact match
    for d in data:
        hyp = d["hypothesis"]
        if hyp not in hyp_to_index:
            dedup_hyp.append(hyp)
            dedup_struct_hyp.append(hyp_dict_to_str(d))
            hyp_to_index[hyp] = len(dedup_struct_hyp) - 1
        orig_to_dedup.append(hyp_to_index[hyp])
    n_dedup = len(dedup_struct_hyp)

    # Generate embeddings for deduplicated hypotheses
    embeds = np.array(get_embedding(dedup_hyp))

    # Initialize assignment structures
    clusters = {i: [i] for i in range(n_dedup)}
    cluster_assignment = {i: i for i in range(n_dedup)}
    hac_to_current = {i: i for i in range(n_dedup)}
    cluster_rep = {i: i for i in range(n_dedup)}

    # Perform HAC over LM embeddings and get the linkage matrix
    linkage_matrix = linkage(embeds, method='ward')

    # Iterate through the linkage matrix to additionally merge clusters based on LLM decisions
    for r, row in enumerate(linkage_matrix):
        breakpoint()
        hac_node_id = n_dedup + r
        left_hac, right_hac = int(row[0]), int(row[1])
        left_current = hac_to_current.get(left_hac)
        right_current = hac_to_current.get(right_hac)
        if left_current is None or right_current is None or left_current == right_current:
            hac_to_current[hac_node_id] = left_current if left_current is not None else right_current
            continue
        rep_left, rep_right = cluster_rep[left_current], cluster_rep[right_current]
        struct_left, struct_right = dedup_struct_hyp[rep_left], dedup_struct_hyp[rep_right]
        # Get the LLM merge decision
        llm_decision = get_llm_merge_decision(
            struct_left, struct_right,
            n_samples=n_samples,
            threshold=merge_threshold
        )

        if llm_decision:
            merged_cluster_id = min(left_current, right_current)
            other_cluster_id = max(left_current, right_current)
            clusters[merged_cluster_id] += clusters[other_cluster_id]
            for idx in clusters[merged_cluster_id]:
                cluster_assignment[idx] = merged_cluster_id
            cluster_rep[merged_cluster_id] = random.choice([rep_left, rep_right])
            del clusters[other_cluster_id]
            del cluster_rep[other_cluster_id]
            hac_to_current[hac_node_id] = merged_cluster_id
        else:
            hac_to_current[hac_node_id] = None

    final_labels = [cluster_assignment[orig_to_dedup[i]] for i in range(len(orig_to_dedup))]

    return final_labels, clusters, cluster_assignment, orig_to_dedup


if __name__ == "__main__":
    parser = ArgParser()
    args = parser.parse_args()
    final_labels, clusters, cluster_assignment, orig_to_dedup = dedupe(nodes_or_json_path=args.in_fpath,
                                                                       n_samples=args.n_samples,
                                                                       merge_threshold=args.merge_threshold,
                                                                       seed=args.seed)
    print("Final Labels:", final_labels)
    print("Clusters:", clusters)
    print("Cluster Assignment:", cluster_assignment)
    print("Original to Deduplicated Mapping:", orig_to_dedup)

    if args.out_fpath is not None:
        # Save the results to the output file
        output_data = {
            "final_labels": final_labels,
            "clusters": clusters,
            "cluster_assignment": cluster_assignment,
            "orig_to_dedup": orig_to_dedup
        }
        with open(args.out_fpath, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to {args.out_fpath}")

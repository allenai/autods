import json
import csv
import argparse
from src.beliefs import BELIEF_MODE_TO_CLS
from src.utils import try_loading_dict


class ArgParser(argparse.ArgumentParser):
    def __init__(self, group=None):
        super().__init__(description='Get surprising nodes from MCTS logs')
        self.add_argument('--in_fpath', type=str, required=True,
                          help='mcts_nodes.json file path or directory containing mcts_node_*.json files')
        self.add_argument('--out_fpath', type=str, required=True, help='output CSV file path')


def nodes_to_csv(nodes_or_json_path, out_fpath):
    from src.mcts_utils import get_nodes  # Import here to avoid circular import issues
    mcts_nodes = get_nodes(nodes_or_json_path)

    csv_list = []
    for node in mcts_nodes:
        if node["level"] in [0, 1]:
            continue
        latest_experiment = None
        latest_programmer = None
        latest_code_executor = None
        latest_analyst = None
        latest_reviewer = None
        csv_node = {}
        for msg in reversed(node["messages"]):
            if not latest_experiment and msg.get("name") in ["user_proxy", "experiment_reviser"]:
                latest_experiment = msg
            elif not latest_programmer and msg.get("name") == "experiment_programmer":
                latest_programmer = msg
            elif not latest_analyst and msg.get("name") in ["experiment_analyst", "experiment_code_analyst"]:
                latest_analyst = msg
            elif not latest_code_executor and msg.get("name") == "code_executor":
                latest_code_executor = msg
            elif not latest_reviewer and msg.get("name") == "experiment_reviewer":
                latest_reviewer = msg

        # Skip if any of these is None
        if not latest_experiment or not latest_programmer or not latest_analyst:
            print(f"[CSV] Skipping node {node['level']}_{node['node_idx']} due to missing messages.")
            continue

        try:
            belief_cls = BELIEF_MODE_TO_CLS[node["prior"]["_type"]]
            prior_distribution = belief_cls.DistributionFormat(**node["prior"])
            posterior_distribution = belief_cls.DistributionFormat(**node["posterior"])
            prior_mean = prior_distribution.get_mean_belief()
            posterior_mean = posterior_distribution.get_mean_belief(prior=prior_distribution)
        except:
            prior_mean = None
            posterior_mean = None

        csv_node['id'] = f"{node['level']}_{node['node_idx']}"
        csv_node['hypothesis'] = node['hypothesis']
        try:
            hypothesis = json.loads(node['hypothesis'])
            csv_node['hypothesis'] = hypothesis['hypothesis']
        except:
            if type(node['hypothesis']) is dict:
                csv_node['hypothesis'] = node['hypothesis']['hypothesis']
            else:
                csv_node['hypothesis'] = node['hypothesis']
        csv_node['is_surprise'] = node.get('surprising', None)
        csv_node['degree_of_surprisal'] = abs(
            posterior_mean - prior_mean) if prior_mean is not None and posterior_mean is not None else None
        csv_node['direction_of_surprisal'] = None
        if prior_mean is not None and posterior_mean is not None:
            csv_node['direction_of_surprisal'] = 'negative' if posterior_mean < prior_mean else (
                'unchanged' if posterior_mean == prior_mean else 'positive')
        csv_node['prior_belief'] = prior_mean
        csv_node['posterior_belief'] = posterior_mean

        try:
            experiment_plan = json.loads(latest_experiment['content'])
            csv_node['experiment_plan'] = f"Objective: {experiment_plan.get('objective', '')}\n" \
                                          f"Steps: {experiment_plan.get('steps', '')}\n" \
                                          f"Deliverables: {experiment_plan.get('deliverables', '')}"
        except Exception:
            csv_node['experiment_plan'] = latest_experiment['content']

        csv_node['analysis'] = try_loading_dict(latest_analyst['content']).get('analysis', '')

        experiment_review = try_loading_dict(latest_reviewer['content'])
        csv_node['review_success'] = experiment_review.get('success', False)
        csv_node['review_feedback'] = experiment_review.get('feedback', 'N/A')

        csv_list.append(csv_node)
    csv_list.sort(key=lambda x: x['degree_of_surprisal'] if x['degree_of_surprisal'] is not None else float('-inf'),
                  reverse=True)

    with open(out_fpath, 'w', newline='') as csv_file:
        fieldnames = ['id', 'hypothesis', 'is_surprise', 'degree_of_surprisal', 'direction_of_surprisal',
                      'prior_belief', 'posterior_belief', 'experiment_plan', 'analysis', 'review_success',
                      'review_feedback']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_list:
            row = {k: (v if v is not None else '') for k, v in row.items()}
            writer.writerow(row)

    print(f"[CSV] MCTS nodes (n={len(csv_list)}; skipping root) saved to {out_fpath}.\n")


if __name__ == '__main__':
    parser = ArgParser()
    args = parser.parse_args()
    nodes_to_csv(args.in_fpath, args.out_fpath)

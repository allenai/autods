import json
import csv
import argparse
from beliefs import BELIEF_MODE_TO_CLS
from utils import get_nodes


class ArgParser(argparse.ArgumentParser):
    def __init__(self, group=None):
        super().__init__(description='Get surprising nodes from MCTS logs')
        self.add_argument('--in_fpath', type=str, required=True,
                          help='mcts_nodes.json file path or directory containing mcts_node_*.json files')
        self.add_argument('--out_fpath', type=str, required=True, help='output CSV file path')


def nodes_to_csv(nodes_or_json_path, out_fpath):
    mcts_nodes = get_nodes(nodes_or_json_path)

    csv_list = []
    for node in mcts_nodes:
        if node["level"] in [0, 1]:
            continue
        latest_experiment = None
        latest_programmer = None
        latest_code_executor = None
        latest_analyst = None
        csv_node = {}
        for msg in reversed(node["messages"]):
            if not latest_experiment and msg.get("name") in ("user_proxy", "experiment_reviser"):
                latest_experiment = msg
            elif not latest_programmer and msg.get("name") == "experiment_programmer":
                latest_programmer = msg
            elif not latest_analyst and msg.get("name") == "experiment_analyst":
                latest_analyst = msg
            elif not latest_code_executor and msg.get("name") == "code_executor":
                latest_code_executor = msg

        try:
            belief_cls = BELIEF_MODE_TO_CLS['categorical']
            prior_distribution = belief_cls.DistributionFormat(**node["prior"])
            posterior_distribution = belief_cls.DistributionFormat(**node["posterior"])
            prior_mean = belief_cls.get_mean_belief(prior_distribution)
            posterior_mean = belief_cls.get_mean_posterior_belief(posterior_distribution, prior_distribution)
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
            csv_node['experiment_plan'] = f"Title: {experiment_plan.get('title', '')}\n" \
                                          f"Objective: {experiment_plan.get('objective', '')}\n" \
                                          f"Steps: {experiment_plan.get('steps', '')}\n" \
                                          f"Deliverables: {experiment_plan.get('deliverables', '')}"
        except Exception:
            csv_node['experiment_plan'] = latest_experiment['content']

        csv_node['analysis'] = json.loads(latest_analyst['content'])['analysis']

        csv_list.append(csv_node)
    csv_list.sort(key=lambda x: x['degree_of_surprisal'] if x['degree_of_surprisal'] is not None else float('-inf'),
                  reverse=True)

    with open(out_fpath, 'w', newline='') as csv_file:
        fieldnames = ['id', 'hypothesis', 'is_surprise', 'degree_of_surprisal', 'direction_of_surprisal',
                      'prior_belief', 'posterior_belief', 'experiment_plan', 'analysis']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_list:
            row = {k: (v if v is not None else '') for k, v in row.items()}
            writer.writerow(row)

    print(f"Collated {len(mcts_nodes)} nodes. Saved to {out_fpath}.")


if __name__ == '__main__':
    parser = ArgParser()
    args = parser.parse_args()
    nodes_to_csv(args.in_fpath, args.out_fpath)

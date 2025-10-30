import argparse


class ArgParser(argparse.ArgumentParser):
    def __init__(self, group=None):
        super().__init__(description='Run AutoDS exploration')

        self.add_argument('--dataset_metadata', type=str, required=True, help='Path to dataset metadata.')
        self.add_argument('--out_dir', type=str, required=True, help='Output directory for logs.')
        self.add_argument("--model", type=str, default="o4-mini",
                          help="LLM to use for all agents (except belief distribution agent).")
        self.add_argument("--belief_model", type=str, default="gpt-4o",
                          help="LLM to use for belief distribution agent.")
        self.add_argument("--user_query", type=str,
                          help="Custom user query to condition experiment generation during exploration.")
        self.add_argument("--temperature", type=float, default=1.0,
                          help="Temperature setting for all agents (except the belief agent). Should be set to None for OpenaAI o-series models.")
        self.add_argument("--belief_temperature", type=float, default=1.0,
                          help="Temperature setting for the belief agent. Should be set to None for OpenaAI o-series models.")
        self.add_argument("--reasoning_effort", type=str, help="Reasoning effort for OpenAI o-series models",
                          choices=['low', 'medium', 'high'], default='medium')
        self.add_argument('--continue_from_dir', type=str,
                          help='Path to logs dir from a previous run to continue exploration from')
        self.add_argument('--continue_from_json', type=str,
                          help='Path to mcts_nodes.json file to continue exploration from')
        self.add_argument('--n_experiments', type=int, help='Number of MCTS iterations (max_iterations)', required=True)
        self.add_argument('--k_experiments', type=int, default=8, help='Branching factor for experiments (>= 1)')
        self.add_argument('--allow_generate_experiments', action=argparse.BooleanOptionalAction, default=True,
                          help='Allow nodes to generate new experiments on-demand')
        self.add_argument('--n_belief_samples', type=int, default=30,
                          help='Number of samples for belief distribution evaluation')
        self.add_argument('--timestamp_dir', action=argparse.BooleanOptionalAction, default=True,
                          help='Create timestamped directory for logs')
        self.add_argument('--exploration_weight', type=float, help='Exploration weight for UCB1 selection method',
                          default=2.0)
        self.add_argument('--dataset_metadata_type', type=str, choices=['dbench', 'blade'], default='dbench',
                          help='Type of dataset metadata format (dbench, blade, or ai2)')
        self.add_argument('--work_dir', type=str, required=True, help='Working directory for agents')
        self.add_argument('--delete_work_dir', action=argparse.BooleanOptionalAction, default=True,
                          help='Delete the work directory after exploration')
        self.add_argument('--beam_width', type=int, default=8, help='Beam width for beam search selection method')
        self.add_argument('--use_beam_search', action=argparse.BooleanOptionalAction, default=False,
                          help='Use beam search selection method')
        self.add_argument("--mcts_selection", type=str,
                          choices=['ucb1', 'beam_search', 'pw', 'pw_all', 'ucb1_recursive'], default='ucb1_recursive',
                          help="Selection method to use in MCTS (UCB1, beam search, progressive widening, progressive widening with all nodes)")
        self.add_argument('--pw_k', type=float, help='Progressive widening constant k', default=1.0)
        self.add_argument('--pw_alpha', type=float, help='Progressive widening exponent alpha', default=0.5)
        self.add_argument('--k_parents', type=int, default=3,
                          help='Number of parent levels to include in prompts (None for all)')
        self.add_argument('--implicit_bayes_posterior', action=argparse.BooleanOptionalAction, default=False,
                          help='Whether to use the belief samples with evidence as the direct posterior or to use a Bayesian update that explicitly combines it with the prior.')
        self.add_argument('--surprisal_width', type=float, default=0.2,
                          help='Minimum difference in mean prior and posterior probabilities required to count as a surprisal.')
        self.add_argument('--belief_mode', type=str,
                          choices=['boolean', 'boolean_cat', 'categorical', 'categorical_numeric', 'gaussian'],
                          default='boolean_cat', help='Belief elicitation mode')
        self.add_argument('--use_binary_reward', action=argparse.BooleanOptionalAction, default=False,
                          help='Use binary reward for MCTS instead of a continuous reward (belief change)')
        self.add_argument('--dedupe', action=argparse.BooleanOptionalAction, default=True,
                          help='Run deduplication after MCTS')
        self.add_argument('--only_save_results', action=argparse.BooleanOptionalAction, default=False,
                          help='Only save results without running MCTS')
        self.add_argument('--experiment_first', action=argparse.BooleanOptionalAction, default=False,
                          help='Generate experiments before hypotheses')
        self.add_argument('--code_timeout', type=int, default=30 * 60,
                          help='Timeout for code execution in seconds')
        self.add_argument('--run_eda', action=argparse.BooleanOptionalAction, default=False,
                          help='Run EDA as part of the initial experiment')
        self.add_argument('--n_warmstart', type=int, default=8,
                          help='Number of initial experiments to run after data loading before using MCTS')
        self.add_argument('--use_online_beliefs', action=argparse.BooleanOptionalAction, default=False,
                          help='Use online beliefs conditioned on past surprisals')
        self.add_argument('--evidence_weight', type=float, default=2.0,
                          help='Weight for the experimental evidence when computing posterior beliefs')
        self.add_argument('--kl_scale', type=float, default=5.0,
                          help='Normalization factor for KL divergence in the reward function')
        self.add_argument('--reward_mode', type=str, choices=['belief', 'kl', 'belief_and_kl'],
                          default='kl', help='Reward mode for MCTS (belief change, KL divergence or both)')
        self.add_argument('--warmstart_experiments', type=str,
                          help='Path to JSON file containing a list of warmstart experiments to run before MCTS')
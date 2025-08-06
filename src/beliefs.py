from typing import List, Dict, Tuple
from pydantic import BaseModel, Field
import tiktoken

from src.mcts import MCTSNode
from src.utils import query_llm


class BeliefTrueFalse:
    class DistributionFormat:
        """
        A distribution of beliefs about the hypothesis using true/false responses (Bernoulli).

        Attributes:
            n: Number of samples used to compute the distribution
            n_true: Number of "true" responses
            n_false: Number of "false" responses
            mean: Mean belief probability (optional, computed if not provided)
            prior_params: Parameters for the prior Beta distribution (alpha, beta)
        """

        def __init__(self,
                     n: int = Field(..., description="Number of samples used to compute the distribution"),
                     n_true: int = Field(..., description='Number of "true" responses'),
                     n_false: int = Field(..., description='Number of "false" responses'),
                     mean: float | None = None,
                     prior_params: Tuple[float, float] = (1.0, 1.0),
                     **kwargs):
            self.n = n
            self.n_true = n_true
            self.n_false = n_false
            self.mean = mean
            self.prior_params = prior_params

        def __repr__(self):
            return f"BeliefTrueFalse.DistributionFormat(n={self.n}, n_true={self.n_true}, n_false={self.n_false})"

        def to_dict(self):
            return {
                "_type": "boolean",
                "n": self.n,
                "n_true": self.n_true,
                "n_false": self.n_false,
                "mean": self.mean,
            }

        def get_mean_belief(self, prior=None) -> float:
            """
            Get the mean of the prior/posterior belief distribution.

            Returns:
                float: The mean belief probability.
            """
            if self.mean is None:
                if prior is None:
                    # Assuming a uniform Beta(1,1) prior, we can compute the mean belief using the Beta distribution
                    self.mean = (self.prior_params[0] + self.n_true) / (self.n + sum(self.prior_params))
                else:
                    # Bayesian update: Beta(n_true + a, n_false + b) where a and b are prior parameters
                    post_alpha = prior.n_true + prior.prior_params[0]
                    # post_beta = prior.n_false + prior.prior_params[1]
                    self.mean = (self.n_true + post_alpha) / (self.n + prior.n + sum(prior.prior_params))
            return self.mean

    class ResponseFormat(BaseModel):
        belief: bool = Field(..., description="Whether the hypothesis is true")

    @staticmethod
    def parse_response(response: List[dict],
                       prior_params: Tuple[float, float] = (1.0, 1.0)) -> 'BeliefTrueFalse.DistributionFormat':
        """
        Parse the response from the LLM into a DistributionFormat.

        Args:
            response (dict): The response from the LLM containing belief counts.
            prior_params (Tuple[float, float]): Parameters for the prior Beta distribution (alpha, beta).

        Returns:
            BeliefTrueFalse.DistributionFormat: Parsed distribution format.
        """
        n, n_true, n_false = 0, 0, 0
        for _res in response:
            n += 1
            if _res["belief"] is True:
                n_true += 1
            else:
                n_false += 1

        return BeliefTrueFalse.DistributionFormat(n=n, n_true=n_true, n_false=n_false, prior_params=prior_params)


class BeliefCategorical:
    score_per_category = {
        "definitely_true": 1.0,
        "partially_true": 0.75,
        "uncertain": 0.5,
        "partially_false": 0.25,
        "definitely_false": 0.0
    }

    class DistributionFormat:
        """
        A distribution of beliefs about the hypothesis using categorical buckets (Categorical).
        Attributes:
            n: Number of samples used to compute the distribution
            definitely_true: Number of "definitely true" responses
            partially_true: Number of "partially true" responses
            uncertain: Number of "uncertain" responses
            partially_false: Number of "partially false" responses
            definitely_false: Number of "definitely false" responses
            mean: Mean belief probability (optional, computed if not provided)
            prior_params: Parameters for the prior Dirichlet distribution (alpha1, alpha2, alpha3, alpha4, alpha5)
        """

        def __init__(self,
                     n: int = Field(..., description="Number of samples used to compute the distribution"),
                     definitely_true: int = Field(..., description='Number of "definitely true" responses'),
                     partially_true: int = Field(..., description='Number of "partially true" responses'),
                     uncertain: int = Field(..., description='Number of "uncertain" responses'),
                     partially_false: int = Field(..., description='Number of "partially false" responses'),
                     definitely_false: int = Field(..., description='Number of "definitely false" responses'),
                     mean: float | None = None,
                     prior_params: Tuple[float, float, float, float, float] = (1.0, 1.0, 1.0, 1.0, 1.0),
                     **kwargs):
            self.n = n
            self.definitely_true = definitely_true
            self.partially_true = partially_true
            self.uncertain = uncertain
            self.partially_false = partially_false
            self.definitely_false = definitely_false
            self.mean = mean
            self.prior_params = prior_params  # Parameters for the prior Dirichlet distribution

        def __repr__(self):
            return (f"BeliefCategorical.DistributionFormat(n={self.n}, definitely_true={self.definitely_true}, "
                    f"partially_true={self.partially_true}, uncertain={self.uncertain}, "
                    f"partially_false={self.partially_false}, definitely_false={self.definitely_false})")

        def to_dict(self):
            return {
                "_type": "categorical",
                "n": self.n,
                "definitely_true": self.definitely_true,
                "partially_true": self.partially_true,
                "uncertain": self.uncertain,
                "partially_false": self.partially_false,
                "definitely_false": self.definitely_false,
                "mean": self.mean,
            }

        def get_mean_belief(self, prior=None) -> float:
            """
            Get the mean of the prior/posterior belief distribution.

            Args:
                prior (BeliefCategorical.DistributionFormat): Prior distribution format object.

            Returns:
                float: The mean belief probability.
            """
            if self.mean is None:
                # Compute the mean belief using the Dirichlet distribution
                if prior is None:
                    mean_per_category = {
                        "definitely_true": (self.definitely_true + self.prior_params[0]) / (
                                self.n + sum(self.prior_params)),
                        "partially_true": (self.partially_true + self.prior_params[1]) / (
                                self.n + sum(self.prior_params)),
                        "uncertain": (self.uncertain + self.prior_params[2]) / (self.n + sum(self.prior_params)),
                        "partially_false": (self.partially_false + self.prior_params[3]) / (
                                self.n + sum(self.prior_params)),
                        "definitely_false": (self.definitely_false + self.prior_params[4]) / (
                                self.n + sum(self.prior_params))
                    }
                else:
                    # Bayesian update
                    mean_per_category = {
                        "definitely_true": (self.definitely_true + prior.definitely_true + prior.prior_params[0]) / (
                                self.n + prior.n + sum(prior.prior_params)),
                        "partially_true": (self.partially_true + prior.partially_true + prior.prior_params[1]) / (
                                self.n + prior.n + sum(prior.prior_params)),
                        "uncertain": (self.uncertain + prior.uncertain + prior.prior_params[2]) / (
                                self.n + prior.n + sum(prior.prior_params)),
                        "partially_false": (self.partially_false + prior.partially_false + prior.prior_params[3]) / (
                                self.n + prior.n + sum(prior.prior_params)),
                        "definitely_false": (self.definitely_false + prior.definitely_false + prior.prior_params[4]) / (
                                self.n + prior.n + sum(prior.prior_params))
                    }
                self.mean = sum(
                    mean_per_category[cat] * BeliefCategorical.score_per_category[cat] for cat in mean_per_category)
            return self.mean

    class ResponseFormat(BaseModel):
        belief: str = Field(..., description="Belief about the hypothesis",
                            choices=["definitely true", "partially true", "uncertain",
                                     "partially false", "definitely false"])

    @staticmethod
    def parse_response(response: List[dict], prior_params: Tuple[float, float, float, float, float] = (
            1.0, 1.0, 1.0, 1.0, 1.0)) -> 'BeliefCategorical.DistributionFormat':
        """
        Parse the response from the LLM into a DistributionFormat.

        Args:
            response (dict): The response from the LLM containing belief counts.
            prior_params (Tuple[float, float, float, float, float]): Parameters for the prior Dirichlet distribution.

        Returns:
            BeliefCategorical.DistributionFormat: Parsed distribution format.
        """
        n = len(response)
        definitely_true = sum(1 for _res in response if _res["belief"] == "definitely true")
        partially_true = sum(1 for _res in response if _res["belief"] == "partially true")
        uncertain = sum(1 for _res in response if _res["belief"] == "uncertain")
        partially_false = sum(1 for _res in response if _res["belief"] == "partially false")
        definitely_false = sum(1 for _res in response if _res["belief"] == "definitely false")

        return BeliefCategorical.DistributionFormat(
            n=n,
            definitely_true=definitely_true,
            partially_true=partially_true,
            uncertain=uncertain,
            partially_false=partially_false,
            definitely_false=definitely_false,
            prior_params=prior_params
        )


class BeliefCategoricalNumeric:
    score_per_category = {
        "0-0.2": 0.0,
        "0.2-0.4": 0.25,
        "0.4-0.6": 0.5,
        "0.6-0.8": 0.75,
        "0.8-1.0": 1.0
    }

    class DistributionFormat:
        """
        A distribution of beliefs about the hypothesis using numerical buckets (Categorical).
        Attributes:
            n: Number of samples used to compute the distribution
            bucket_02: Number of responses that fall in the range [0.0, 0.2)
            bucket_24: Number of responses that fall in the range [0.2, 0.4)
            bucket_46: Number of responses that fall in the range [0.4, 0.6)
            bucket_68: Number of responses that fall in the range [0.6, 0.8)
            bucket_810: Number of responses that fall in the range [0.8, 1.0)
            mean: Mean belief probability (optional, computed if not provided)
            prior_params: Parameters for the prior Dirichlet distribution (alpha1, alpha2, alpha3, alpha4, alpha5)
        """

        def __init__(self,
                     n: int = Field(..., description="Number of samples used to compute the distribution"),
                     bucket_02: int = Field(..., description='Number of responses that fall in the range [0.0, 0.2)'),
                     bucket_24: int = Field(..., description='Number of responses that fall in the range [0.2, 0.4)'),
                     bucket_46: int = Field(..., description='Number of responses that fall in the range [0.4, 0.6)'),
                     bucket_68: int = Field(..., description='Number of responses that fall in the range [0.6, 0.8)'),
                     bucket_810: int = Field(..., description='Number of responses that fall in the range [0.8, 1.0)'),
                     mean: float | None = None,
                     prior_params: Tuple[float, float, float, float, float] = (1.0, 1.0, 1.0, 1.0, 1.0),
                     **kwargs):
            self.n = n
            self.bucket_02 = bucket_02
            self.bucket_24 = bucket_24
            self.bucket_46 = bucket_46
            self.bucket_68 = bucket_68
            self.bucket_810 = bucket_810
            self.mean = mean
            self.prior_params = prior_params  # Parameters for the prior Dirichlet distribution

        def __repr__(self):
            return (f"BeliefCategoricalNumeric.DistributionFormat(n={self.n}, bucket_02={self.bucket_02}, "
                    f"bucket_24={self.bucket_24}, bucket_46={self.bucket_46}, "
                    f"bucket_68={self.bucket_68}, bucket_810={self.bucket_810})")

        def to_dict(self):
            return {
                "_type": "categorical_numeric",
                "n": self.n,
                "bucket_02": self.bucket_02,
                "bucket_24": self.bucket_24,
                "bucket_46": self.bucket_46,
                "bucket_68": self.bucket_68,
                "bucket_810": self.bucket_810,
                "mean": self.mean,
            }

        def get_mean_belief(self, prior=None) -> float:
            """
            Get the mean of the prior/posterior belief distribution.

            Args:
                prior (BeliefCategoricalNumeric.DistributionFormat): Prior distribution format object.

            Returns:
                float: The mean belief probability.
            """
            if self.mean is None:
                # Compute the mean belief using the Dirichlet distribution
                if prior is None:
                    mean_per_category = {
                        "0-0.2": (self.bucket_02 + self.prior_params[0]) / (self.n + sum(self.prior_params)),
                        "0.2-0.4": (self.bucket_24 + self.prior_params[1]) / (self.n + sum(self.prior_params)),
                        "0.4-0.6": (self.bucket_46 + self.prior_params[2]) / (self.n + sum(self.prior_params)),
                        "0.6-0.8": (self.bucket_68 + self.prior_params[3]) / (self.n + sum(self.prior_params)),
                        "0.8-1.0": (self.bucket_810 + self.prior_params[4]) / (self.n + sum(self.prior_params))
                    }
                else:
                    # Bayesian update
                    mean_per_category = {
                        "0-0.2": (self.bucket_02 + prior.bucket_02 + prior.prior_params[0]) / (
                                self.n + prior.n + sum(prior.prior_params)),
                        "0.2-0.4": (self.bucket_24 + prior.bucket_24 + prior.prior_params[1]) / (
                                self.n + prior.n + sum(prior.prior_params)),
                        "0.4-0.6": (self.bucket_46 + prior.bucket_46 + prior.prior_params[2]) / (
                                self.n + prior.n + sum(prior.prior_params)),
                        "0.6-0.8": (self.bucket_68 + prior.bucket_68 + prior.prior_params[3]) / (
                                self.n + prior.n + sum(prior.prior_params)),
                        "0.8-1.0": (self.bucket_810 + prior.bucket_810 + prior.prior_params[4]) / (
                                self.n + prior.n + sum(prior.prior_params))
                    }
                self.mean = sum(mean_per_category[cat] * BeliefCategoricalNumeric.score_per_category[cat] for cat in
                                mean_per_category)

            return self.mean

    class ResponseFormat(BaseModel):
        belief: str = Field(..., description="Belief about the hypothesis being true",
                            choices=["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"])

    @staticmethod
    def parse_response(response: List[dict], prior_params: Tuple[float, float, float, float, float] = (
            1.0, 1.0, 1.0, 1.0, 1.0)) -> 'BeliefCategoricalNumeric.DistributionFormat':
        """
        Parse the response from the LLM into a DistributionFormat.

        Args:
            response (dict): The response from the LLM containing belief counts.
            prior_params (Tuple[float, float, float, float, float]): Parameters for the prior Dirichlet distribution.

        Returns:
            BeliefCategoricalNumeric.DistributionFormat: Parsed distribution format.
        """
        n = len(response)
        bucket_02 = sum(1 for _res in response if _res["belief"] == "0-0.2")
        bucket_24 = sum(1 for _res in response if _res["belief"] == "0.2-0.4")
        bucket_46 = sum(1 for _res in response if _res["belief"] == "0.4-0.6")
        bucket_68 = sum(1 for _res in response if _res["belief"] == "0.6-0.8")
        bucket_810 = sum(1 for _res in response if _res["belief"] == "0.8-1.0")

        return BeliefCategoricalNumeric.DistributionFormat(
            n=n,
            bucket_02=bucket_02,
            bucket_24=bucket_24,
            bucket_46=bucket_46,
            bucket_68=bucket_68,
            bucket_810=bucket_810,
            prior_params=prior_params
        )


class BeliefProb:
    """
    A distribution of beliefs about the hypothesis using samples of the mean probability directly (Gaussian)

    Attributes:
        n: Number of samples used to compute the distribution
        mean: Mean probability of the hypothesis being true
        stddev: Standard deviation of the probabilities
    """

    class DistributionFormat:
        def __init__(self,
                     n: int = Field(..., description="Number of samples used to compute the distribution"),
                     mean: float = Field(..., description="Mean probability of the hypothesis being true"),
                     stddev: float = Field(..., description="Standard deviation of the probabilities"),
                     **kwargs):
            self.n = n
            self.mean = mean
            self.stddev = stddev

        def __repr__(self):
            return f"BeliefProb.DistributionFormat(n={self.n}, mean={self.mean}, stddev={self.stddev})"

        def to_dict(self):
            return {
                "_type": "probability",
                "n": self.n,
                "mean": self.mean,
                "stddev": self.stddev
            }

        def get_mean_belief(self, prior=None) -> float:
            """
            Get the mean of the prior/posterior belief distribution.

            Args:
                prior (BeliefProb.DistributionFormat): Prior distribution format object.

            Returns:
                float: The mean belief probability.
            """
            if prior is not None:
                # Bayesian update: Estimate Beta parameters from the mean and stddev for the prior and posterior distributions
                prior_mean = prior.mean
                prior_var = prior.stddev ** 2
                prior_alpha = prior_mean * (prior_mean * (1 - prior_mean) / prior_var - 1)
                prior_beta = (1 - prior_mean) * (prior_mean * (1 - prior_mean) / prior_var - 1)
                like_mean = self.mean
                like_var = self.stddev ** 2
                like_alpha = like_mean * (like_mean * (1 - like_mean) / like_var - 1)
                like_beta = (1 - like_mean) * (like_mean * (1 - like_mean) / like_var - 1)
                # Posterior parameters
                post_alpha = prior_alpha + like_alpha - 1  # -1 because we are using the mean of the likelihood
                post_beta = prior_beta + like_beta - 1  # -1 because we are using the mean of the likelihood
                # Mean of the posterior Beta distribution
                self.mean = post_alpha / (post_alpha + post_beta)
                # Also compute the standard deviation of the posterior
                self.stddev = (post_alpha * post_beta) / (
                        (post_alpha + post_beta) ** 2 * (post_alpha + post_beta + 1)) ** 0.5

            return self.mean

    class ResponseFormat(BaseModel):
        belief: float = Field(..., description="Mean probability of the hypothesis being true",
                              ge=0.0, le=1.0)

    @staticmethod
    def parse_response(response: List[dict]) -> 'BeliefProb.DistributionFormat':
        """
        Parse the response from the LLM into a DistributionFormat.

        Args:
            response (dict): The response from the LLM containing belief probabilities.

        Returns:
            BeliefProb.DistributionFormat: Parsed distribution format.
        """
        n = len(response)
        mean = sum(_res["belief"] for _res in response) / n
        stddev = (sum((_res["belief"] - mean) ** 2 for _res in response) / n) ** 0.5

        return BeliefProb.DistributionFormat(n=n, mean=mean, stddev=stddev)


BELIEF_MODE_TO_CLS = {
    "boolean": BeliefTrueFalse,
    "categorical": BeliefCategorical,
    "categorical_numeric": BeliefCategoricalNumeric,
    "probability": BeliefProb
}


def get_belief(
        hypothesis: str,
        evidence: List[Dict[str, str]] = None,
        model: str = "gpt-4o",
        belief_mode: str = "boolean",
        n_samples: int = 5,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        use_prior: bool = False,
        explicit_prior=None,
        n_retries=3
):
    """
    Get belief distribution for a hypothesis with optional evidence.

    Args:
        hypothesis: The hypothesis to evaluate
        evidence: Optional evidence messages to condition the belief
        model: The LLM model to use
        belief_mode: The belief mode to use for parsing responses (e.g., BeliefTrueFalse, BeliefCategorical)
        n_samples: Number of samples to draw from the LLM
        temperature: Temperature for sampling
        reasoning_effort: Reasoning effort for o-series models
        use_prior: Whether to use implicit Bayesian posterior
        explicit_prior: Optional prior distribution to use for Bayesian updates
        n_retries: Number of retries for querying the LLM in case of errors
    """
    belief_cls = BELIEF_MODE_TO_CLS.get(belief_mode)

    # Construct the system prompt based on whether we are eliciting prior, implicit posterior, or explicit posterior beliefs
    _system_msgs = [
        "You are a research scientist whose task is to judge whether the given hypothesis holds true or not."
    ]
    if evidence is not None:
        # posterior belief
        _system_msgs.append(
            "Use the provided evidence collected from running a scientific experiment (i.e., the experiment plan, program, execution output, and analysis) to help in making a judgement."
        )
    # else:  # prior belief
    if use_prior:
        # implicit posterior
        _system_msgs.append(
            "Use your prior knowledge of the research domain to help in your assessment."
        )
    else:
        # explicit posterior
        assert evidence is not None
        _system_msgs.append(
            "Disregard your prior beliefs about the research domain and focus only on the provided evidence."
        )
    system_prompt = {
        "role": "system",
        "content": "".join(_system_msgs)
    }

    hypothesis_msg = {
        "role": "user",
        "content": f"Hypothesis: {hypothesis}\n\nCarefully reason before making your assessment."
    }

    all_msgs = [system_prompt]
    if evidence is not None:
        all_msgs += evidence
    all_msgs.append(hypothesis_msg)

    for attempt in range(n_retries):
        try:
            response = query_llm(all_msgs, model=model, n_samples=n_samples,
                                 temperature=temperature, reasoning_effort=reasoning_effort,
                                 response_format=belief_cls.ResponseFormat)
            distribution = belief_cls.parse_response(response)

            # If we are using an explicit prior, we need to combine it with the posterior distribution
            if not use_prior and explicit_prior is not None:
                mean_belief = distribution.get_mean_belief(prior=explicit_prior)
            else:
                mean_belief = distribution.get_mean_belief()
        except Exception as e:
            if attempt == n_retries - 1:
                print(f"Querying LLM: ERROR: {e}\nMax retries reached. Returning empty distribution.")
                return None, None
            else:
                print(f"Querying LLM: ERROR: {e}\nRetrying ({attempt + 1}/{n_retries})...")

    return distribution, mean_belief


def calculate_prior_and_posterior_beliefs(node, n_samples=4, model="gpt-4o", temperature=None,
                                          reasoning_effort=None, implicit_bayes_posterior=False, surprisal_width=0.2,
                                          belief_mode="boolean"):
    """
    Calculate prior and posterior belief distributions for a hypothesis.

    Args:
        node: MCTSNode instance containing node information and messages or a dictionary with node data
    """

    MODEL_CTXT_LIMITS = {
        "o4-mini": 200_000,
        "gpt-4o": 128_000,
    }

    if type(node) is MCTSNode:
        hypothesis = node.hypothesis
        if type(hypothesis) is dict:
            hypothesis = hypothesis["hypothesis"]
        messages = node.messages or []
    else:
        hypothesis = node.get("hypothesis", None)
        if type(hypothesis) is dict:
            hypothesis = hypothesis["hypothesis"]
        messages = node.get("messages", [])

    # Only include the latest experiment plan or experiment reviser, programmer, and analyst messages
    latest_experiment = None
    latest_programmer = None
    latest_code_executor = None
    latest_analyst = None

    for msg in reversed(messages):
        if not latest_experiment and msg.get("name") in ["user_proxy", "experiment_reviser"]:
            latest_experiment = msg
        elif not latest_programmer and msg.get("name") == "experiment_programmer":
            latest_programmer = msg
        elif not latest_analyst and msg.get("name") in ["experiment_analyst", "experiment_code_analyst"]:
            latest_analyst = msg
        elif not latest_code_executor and msg.get("name") == "code_executor":
            latest_code_executor = msg
            # Make sure the input tokens do not exceed 200,000 tokens
            encoding = tiktoken.encoding_for_model("gpt-4o")  # Currently, "o4-mini" raises an error
            input_tokens = len(encoding.encode(latest_code_executor["content"]))
            if input_tokens > (MODEL_CTXT_LIMITS[model] - 10_000):  # Reserve 10k tokens for all other messages
                # Truncate the content of the code executor from the beginning
                latest_code_executor["content"] = encoding.decode(
                    encoding.encode(latest_code_executor["content"])[-int(MODEL_CTXT_LIMITS[model] - 10_000):]
                )

        if latest_experiment and latest_programmer and latest_analyst and latest_code_executor:
            break

    evidence = [m for m in [latest_experiment, latest_programmer, latest_code_executor, latest_analyst] if m]

    prior, prior_mean = get_belief(
        hypothesis=hypothesis,
        evidence=None,
        model=model,
        belief_mode=belief_mode,
        n_samples=n_samples,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        use_prior=True,
    )
    posterior, posterior_mean = get_belief(
        hypothesis=hypothesis,
        evidence=evidence,
        model=model,
        belief_mode=belief_mode,
        n_samples=n_samples,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        use_prior=implicit_bayes_posterior,
        explicit_prior=prior
    )

    if prior is None or posterior is None:
        return None, None, None, None

    belief_change = abs(posterior_mean - prior_mean)
    is_surprisal = belief_change >= surprisal_width
    # is_surprisal_in_diff_02buckets = (prior_mean*10)//2 != (posterior_mean*10)//2

    return is_surprisal, belief_change, prior, posterior

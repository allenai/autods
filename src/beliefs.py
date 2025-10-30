from typing import List, Dict, Tuple, Optional

import numpy as np
from pydantic import BaseModel, Field

from src.mcts import MCTSNode
from src.mcts_utils import get_context_string
from src.utils import query_llm, fuse_gaussians

from scipy.special import betaln, gammaln, psi  # betaln = log Beta function, psi = digamma


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
                     n: float = Field(..., description="Number of samples used to compute the distribution"),
                     n_true: float = Field(..., description='Number of "true" responses'),
                     n_false: float = Field(..., description='Number of "false" responses'),
                     mean: float | None = None,
                     prior_params: Tuple[float, float] = (0.5, 0.5),
                     **kwargs):
            self.n = n
            self.n_true = n_true
            self.n_false = n_false
            self.mean = mean
            self._empirical_mean = 0.5
            self.prior_params = prior_params

        def __repr__(self):
            return f"BeliefTrueFalse.DistributionFormat(n={self.n}, n_true={self.n_true}, n_false={self.n_false})"

        def to_dict(self):
            return {
                "_type": "boolean",
                "prior_params": self.prior_params,
                "n": self.n,
                "n_true": self.n_true,
                "n_false": self.n_false,
                "_empirical_mean": self._empirical_mean,
                "mean": self.mean,
            }

        def get_mean_belief(self, prior=None, recompute=False) -> float:
            """
            Get the mean of the prior/posterior belief distribution.
            Args:
                prior (BeliefTrueFalse.DistributionFormat): Prior distribution format object.
                recompute (bool): Whether to recompute the mean even if it is already set.
            Returns:
                float: The mean belief probability.
            """
            if self.mean is None or recompute:
                # Compute the mean belief using the Beta distribution
                if self.n > 0:
                    self._empirical_mean = self.n_true / self.n
                self.mean = (self.prior_params[0] + self.n_true) / (self.n + sum(self.prior_params))

                if prior is not None:
                    # Bayesian update: Beta(n_true + a, n_false + b) where a and b are prior parameters
                    post_alpha = prior.n_true + prior.prior_params[0]
                    # post_beta = prior.n_false + prior.prior_params[1]
                    self.mean = (self.n_true + post_alpha) / (self.n + prior.n + sum(prior.prior_params))
            return self.mean

        def update(self, n_true: int = 0, n_false: int = 0, distr=None, normalize: bool = False):
            """
            Update the distribution with new counts.
            """
            if distr is not None:
                self.n_true += distr.n_true
                self.n_false += distr.n_false
            else:
                self.n_true += n_true
                self.n_false += n_false
            n = distr.n if distr is not None else (n_true + n_false)
            if normalize:
                total = self.n + n
                self.n_true /= (total / self.n)
                self.n_false /= (total / self.n)
            else:
                self.n += n
            # Reset mean
            _ = self.get_mean_belief(recompute=True)

        def get_params(self) -> Tuple[float, float]:
            """
            Get the parameters of the Beta distribution.
            Returns:
                Tuple[float, float]: Parameters (alpha, beta) of the Beta distribution.
            """
            return self.prior_params[0] + self.n_true, self.prior_params[1] + self.n_false

    class ResponseFormat(BaseModel):
        """
        Belief about the support for the hypothesis.

        Attributes:
            belief (bool | None): Whether the hypothesis is true or false. If you do not have enough information to
            comment on the hypothesis, return None.
        """
        belief: bool | None = Field(..., description="Whether the hypothesis is true")

    @staticmethod
    def parse_response(response: List[dict],
                       prior_params: Tuple[float, float] = (0.5, 0.5),
                       weight: float = 1.0) -> 'BeliefTrueFalse.DistributionFormat':
        """
        Parse the response from the LLM into a DistributionFormat.

        Args:
            response (dict): The response from the LLM containing belief counts.
            prior_params (Tuple[float, float]): Parameters for the prior Beta distribution (alpha, beta).
            weight (float): Weight to apply to the counts (default is 1.0).

        Returns:
            BeliefTrueFalse.DistributionFormat: Parsed distribution format.
        """
        n, n_true, n_false = 0, 0, 0
        for _res in response:
            if _res["belief"] is not None:
                n += 1  # Count only responses that provide a belief
                n_true += int(_res["belief"])
                n_false += int(not _res["belief"])
        n *= weight
        n_true *= weight
        n_false *= weight

        return BeliefTrueFalse.DistributionFormat(n=n, n_true=n_true, n_false=n_false, prior_params=prior_params)

    @staticmethod
    def kl_divergence(distr1: 'BeliefTrueFalse.DistributionFormat',
                      distr2: 'BeliefTrueFalse.DistributionFormat') -> float:
        """
        Compute the KL divergence between two distributions.
        Args:
            distr1 (BeliefTrueFalse.DistributionFormat): First distribution.
            distr2 (BeliefTrueFalse.DistributionFormat): Second distribution.
        Returns:
            float: KL divergence between the two distributions.
        """
        alpha1, beta1 = distr1.get_params()
        alpha2, beta2 = distr2.get_params()
        term1 = betaln(alpha2, beta2) - betaln(alpha1, beta1)
        term2 = (alpha1 - alpha2) * psi(alpha1)
        term3 = (beta1 - beta2) * psi(beta1)
        term4 = (alpha2 - alpha1 + beta2 - beta1) * psi(alpha1 + beta1)
        return term1 + term2 + term3 + term4


class BeliefCategorical:
    score_per_category = {
        "definitely_false": 0.1,
        "maybe_false": 0.3,
        "uncertain": 0.5,
        "maybe_true": 0.7,
        "definitely_true": 0.9
    }

    class DistributionFormat:
        """
        A distribution of beliefs about the hypothesis using categorical buckets (Categorical).
        Attributes:
            n: Number of samples used to compute the distribution
            definitely_true: Number of "definitely true" responses
            maybe_true: Number of "maybe true" responses
            uncertain: Number of "uncertain" responses
            maybe_false: Number of "maybe false" responses
            definitely_false: Number of "definitely false" responses
            mean: Mean belief probability (optional, computed if not provided)
            prior_params: Parameters for the prior Dirichlet distribution (alpha1, alpha2, alpha3, alpha4, alpha5)
        """

        def __init__(self,
                     n: float = Field(..., description="Number of samples used to compute the distribution"),
                     definitely_true: float = Field(..., description='Number of "definitely true" responses'),
                     maybe_true: float = Field(..., description='Number of "maybe true" responses'),
                     uncertain: float = Field(..., description='Number of "uncertain" responses'),
                     maybe_false: float = Field(..., description='Number of "maybe false" responses'),
                     definitely_false: float = Field(..., description='Number of "definitely false" responses'),
                     mean: float | None = None,
                     prior_params: Tuple[float, float, float, float, float] = (0.2, 0.2, 0.2, 0.2, 0.2),
                     **kwargs):
            self.n = n
            self.definitely_true = definitely_true
            self.maybe_true = maybe_true
            self.uncertain = uncertain
            self.maybe_false = maybe_false
            self.definitely_false = definitely_false
            self.mean = mean
            self._empirical_mean = 0.5
            self.prior_params = prior_params  # Parameters for the prior Dirichlet distribution

        def __repr__(self):
            return (f"BeliefCategorical.DistributionFormat(n={self.n}, definitely_true={self.definitely_true}, "
                    f"maybe_true={self.maybe_true}, uncertain={self.uncertain}, "
                    f"maybe_false={self.maybe_false}, definitely_false={self.definitely_false})")

        def to_dict(self):
            return {
                "_type": "categorical",
                "prior_params": self.prior_params,
                "n": self.n,
                "definitely_true": self.definitely_true,
                "maybe_true": self.maybe_true,
                "uncertain": self.uncertain,
                "maybe_false": self.maybe_false,
                "definitely_false": self.definitely_false,
                "_empirical_mean": self._empirical_mean,
                "mean": self.mean,
            }

        def get_mean_belief(self, prior=None, recompute=False) -> float:
            """
            Get the mean of the prior/posterior belief distribution.
            Args:
                prior (BeliefCategorical.DistributionFormat): Prior distribution format object.
                recompute (bool): Whether to recompute the mean even if it is already set.
            Returns:
                float: The mean belief probability.
            """
            if self.mean is None or recompute:
                # Compute the mean belief using the Dirichlet distribution
                if self.n > 0:
                    mean_per_category = {
                        "definitely_true": self.definitely_true / self.n,
                        "maybe_true": self.maybe_true / self.n,
                        "uncertain": self.uncertain / self.n,
                        "maybe_false": self.maybe_false / self.n,
                        "definitely_false": self.definitely_false / self.n
                    }
                    self._empirical_mean = sum(
                        mean_per_category[cat] * BeliefCategorical.score_per_category[cat] for cat in mean_per_category)

                mean_per_category = {
                    "definitely_true": (self.definitely_true + self.prior_params[0]) / (
                            self.n + sum(self.prior_params)),
                    "maybe_true": (self.maybe_true + self.prior_params[1]) / (
                            self.n + sum(self.prior_params)),
                    "uncertain": (self.uncertain + self.prior_params[2]) / (self.n + sum(self.prior_params)),
                    "maybe_false": (self.maybe_false + self.prior_params[3]) / (
                            self.n + sum(self.prior_params)),
                    "definitely_false": (self.definitely_false + self.prior_params[4]) / (
                            self.n + sum(self.prior_params))
                }
                self.mean = sum(
                    mean_per_category[cat] * BeliefCategorical.score_per_category[cat] for cat in mean_per_category)

                if prior is not None:
                    # Bayesian update
                    mean_per_category = {
                        "definitely_true": (self.definitely_true + prior.definitely_true + prior.prior_params[0]) / (
                                self.n + prior.n + sum(prior.prior_params)),
                        "maybe_true": (self.maybe_true + prior.maybe_true + prior.prior_params[1]) / (
                                self.n + prior.n + sum(prior.prior_params)),
                        "uncertain": (self.uncertain + prior.uncertain + prior.prior_params[2]) / (
                                self.n + prior.n + sum(prior.prior_params)),
                        "maybe_false": (self.maybe_false + prior.maybe_false + prior.prior_params[3]) / (
                                self.n + prior.n + sum(prior.prior_params)),
                        "definitely_false": (self.definitely_false + prior.definitely_false + prior.prior_params[4]) / (
                                self.n + prior.n + sum(prior.prior_params))
                    }
                    self.mean = sum(
                        mean_per_category[cat] * BeliefCategorical.score_per_category[cat] for cat in mean_per_category)
            return self.mean

        def update(self,
                   definitely_true: int = 0,
                   maybe_true: int = 0,
                   uncertain: int = 0,
                   maybe_false: int = 0,
                   definitely_false: int = 0,
                   distr=None,
                   normalize: bool = False):
            """
            Update the distribution with new counts.
            """
            if distr is not None:
                self.definitely_true += distr.definitely_true
                self.maybe_true += distr.maybe_true
                self.uncertain += distr.uncertain
                self.maybe_false += distr.maybe_false
                self.definitely_false += distr.definitely_false
            else:
                self.definitely_true += definitely_true
                self.maybe_true += maybe_true
                self.uncertain += uncertain
                self.maybe_false += maybe_false
                self.definitely_false += definitely_false
            n = distr.n if distr is not None else (
                    definitely_true + maybe_true + uncertain + maybe_false + definitely_false
            )
            if normalize:
                total = self.n + n
                self.definitely_true /= (total / self.n)
                self.maybe_true /= (total / self.n)
                self.uncertain /= (total / self.n)
                self.maybe_false /= (total / self.n)
                self.definitely_false /= (total / self.n)
            else:
                self.n += n
            # Reset mean
            _ = self.get_mean_belief(recompute=True)

        def get_params(self) -> Tuple[float, float, float, float, float]:
            """
            Get the parameters of the Dirichlet distribution.
            Returns:
                Tuple[float, float, float, float, float]: Parameters (alpha1, alpha2, alpha3, alpha4, alpha5) of the Dirichlet distribution.
            """
            return (self.prior_params[0] + self.definitely_true,
                    self.prior_params[1] + self.maybe_true,
                    self.prior_params[2] + self.uncertain,
                    self.prior_params[3] + self.maybe_false,
                    self.prior_params[4] + self.definitely_false)

    class ResponseFormat(BaseModel):
        """
        Belief about the support for the hypothesis.

        Attributes:
            belief (str): Belief about the support for the hypothesis. Choices are:
                "definitely true": Hypothesis is definitely true.
                "maybe true": Hypothesis may be true.
                "uncertain": Hypothesis is equally likely to be true or false (e.g., because of relevant but contradictory evidence).
                "maybe false": Hypothesis may be false.
                "definitely false": Hypothesis is definitely false.
                "cannot comment": Not enough information to comment on the hypothesis (e.g., due to lack of domain knowledge or lack of relevant evidence).
        """
        belief: str = Field(..., description="Belief about the hypothesis",
                            choices=["definitely true", "maybe true", "uncertain",
                                     "maybe false", "definitely false", "cannot comment"])

    @staticmethod
    def parse_response(response: List[dict], prior_params: Tuple[float, float, float, float, float] = (
            0.2, 0.2, 0.2, 0.2, 0.2), weight=1.0) -> 'BeliefCategorical.DistributionFormat':
        """
        Parse the response from the LLM into a DistributionFormat.

        Args:
            response (dict): The response from the LLM containing belief counts.
            prior_params (Tuple[float, float, float, float, float]): Parameters for the prior Dirichlet distribution.
            weight (float): Weight to apply to the counts (default is 1.0).

        Returns:
            BeliefCategorical.DistributionFormat: Parsed distribution format.
        """
        cannot_comment = sum(1 for _res in response if _res["belief"] == "cannot comment")
        definitely_true = weight * sum(1 for _res in response if _res["belief"] == "definitely true")
        maybe_true = weight * sum(1 for _res in response if _res["belief"] == "maybe true")
        uncertain = weight * sum(1 for _res in response if _res["belief"] == "uncertain")
        maybe_false = weight * sum(1 for _res in response if _res["belief"] == "maybe false")
        definitely_false = weight * sum(1 for _res in response if _res["belief"] == "definitely false")
        n = weight * (len(response) - cannot_comment)  # Exclude responses with "cannot comment"

        return BeliefCategorical.DistributionFormat(
            n=n,
            definitely_true=definitely_true,
            maybe_true=maybe_true,
            uncertain=uncertain,
            maybe_false=maybe_false,
            definitely_false=definitely_false,
            prior_params=prior_params
        )

    @staticmethod
    def kl_divergence(distr1: 'BeliefCategorical.DistributionFormat',
                      distr2: 'BeliefCategorical.DistributionFormat') -> float:
        """
        Compute the KL divergence between two distributions.
        Args:
            distr1 (BeliefCategorical.DistributionFormat): First distribution.
            distr2 (BeliefCategorical.DistributionFormat): Second distribution.
        Returns:
            float: KL divergence between the two distributions.
        """
        alpha = np.array(distr1.get_params())
        beta = np.array(distr2.get_params())

        sum_alpha = np.sum(alpha)
        sum_beta = np.sum(beta)

        term1 = gammaln(sum_alpha) - np.sum(gammaln(alpha))
        term2 = -gammaln(sum_beta) + np.sum(gammaln(beta))
        term3 = np.sum((alpha - beta) * (psi(alpha) - psi(sum_alpha)))

        return term1 + term2 + term3


class BeliefCategoricalNumeric:
    score_per_category = {
        "0-0.2": 0.1,
        "0.2-0.4": 0.3,
        "0.4-0.6": 0.5,
        "0.6-0.8": 0.7,
        "0.8-1.0": 0.9
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
                     n: float = Field(..., description="Number of samples used to compute the distribution"),
                     bucket_02: float = Field(..., description='Number of responses that fall in the range [0.0, 0.2)'),
                     bucket_24: float = Field(..., description='Number of responses that fall in the range [0.2, 0.4)'),
                     bucket_46: float = Field(..., description='Number of responses that fall in the range [0.4, 0.6)'),
                     bucket_68: float = Field(..., description='Number of responses that fall in the range [0.6, 0.8)'),
                     bucket_810: float = Field(...,
                                               description='Number of responses that fall in the range [0.8, 1.0)'),
                     mean: float | None = None,
                     prior_params: Tuple[float, float, float, float, float] = (0.2, 0.2, 0.2, 0.2, 0.2),
                     **kwargs):
            self.n = n
            self.bucket_02 = bucket_02
            self.bucket_24 = bucket_24
            self.bucket_46 = bucket_46
            self.bucket_68 = bucket_68
            self.bucket_810 = bucket_810
            self.mean = mean
            self._empirical_mean = 0.5
            self.prior_params = prior_params  # Parameters for the prior Dirichlet distribution

        def __repr__(self):
            return (f"BeliefCategoricalNumeric.DistributionFormat(n={self.n}, bucket_02={self.bucket_02}, "
                    f"bucket_24={self.bucket_24}, bucket_46={self.bucket_46}, "
                    f"bucket_68={self.bucket_68}, bucket_810={self.bucket_810})")

        def to_dict(self):
            return {
                "_type": "categorical_numeric",
                "prior_params": self.prior_params,
                "n": self.n,
                "bucket_02": self.bucket_02,
                "bucket_24": self.bucket_24,
                "bucket_46": self.bucket_46,
                "bucket_68": self.bucket_68,
                "bucket_810": self.bucket_810,
                "_empirical_mean": self._empirical_mean,
                "mean": self.mean,
            }

        def get_mean_belief(self, prior=None, recompute=False) -> float:
            """
            Get the mean of the prior/posterior belief distribution.
            Args:
                prior (BeliefCategoricalNumeric.DistributionFormat): Prior distribution format object.
                recompute (bool): Whether to recompute the mean even if it is already set.
            Returns:
                float: The mean belief probability.
            """
            if self.mean is None or recompute:
                # Compute the mean belief using the Dirichlet distribution
                if self.n > 0:
                    mean_per_category = {
                        "0-0.2": self.bucket_02 / self.n,
                        "0.2-0.4": self.bucket_24 / self.n,
                        "0.4-0.6": self.bucket_46 / self.n,
                        "0.6-0.8": self.bucket_68 / self.n,
                        "0.8-1.0": self.bucket_810 / self.n
                    }
                    self._empirical_mean = sum(
                        mean_per_category[cat] * BeliefCategoricalNumeric.score_per_category[cat] for cat in
                        mean_per_category)

                mean_per_category = {
                    "0-0.2": (self.bucket_02 + self.prior_params[0]) / (self.n + sum(self.prior_params)),
                    "0.2-0.4": (self.bucket_24 + self.prior_params[1]) / (self.n + sum(self.prior_params)),
                    "0.4-0.6": (self.bucket_46 + self.prior_params[2]) / (self.n + sum(self.prior_params)),
                    "0.6-0.8": (self.bucket_68 + self.prior_params[3]) / (self.n + sum(self.prior_params)),
                    "0.8-1.0": (self.bucket_810 + self.prior_params[4]) / (self.n + sum(self.prior_params))
                }
                self.mean = sum(
                    mean_per_category[cat] * BeliefCategoricalNumeric.score_per_category[cat] for cat in
                    mean_per_category)

                if prior is not None:
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

        def update(self,
                   bucket_02: int = 0,
                   bucket_24: int = 0,
                   bucket_46: int = 0,
                   bucket_68: int = 0,
                   bucket_810: int = 0,
                   distr=None,
                   normalize: bool = False):
            """
            Update the distribution with new counts.
            """
            if distr is not None:
                self.bucket_02 += distr.bucket_02
                self.bucket_24 += distr.bucket_24
                self.bucket_46 += distr.bucket_46
                self.bucket_68 += distr.bucket_68
                self.bucket_810 += distr.bucket_810
            else:
                self.bucket_02 += bucket_02
                self.bucket_24 += bucket_24
                self.bucket_46 += bucket_46
                self.bucket_68 += bucket_68
                self.bucket_810 += bucket_810
            n = distr.n if distr is not None else (
                    bucket_02 + bucket_24 + bucket_46 + bucket_68 + bucket_810
            )
            if normalize:
                total = self.n + n
                self.bucket_02 /= (total / self.n)
                self.bucket_24 /= (total / self.n)
                self.bucket_46 /= (total / self.n)
                self.bucket_68 /= (total / self.n)
                self.bucket_810 /= (total / self.n)
            else:
                self.n += n
            # Reset mean
            _ = self.get_mean_belief(recompute=True)

        def get_params(self) -> Tuple[float, float, float, float, float]:
            """
            Get the parameters of the Dirichlet distribution.
            Returns:
                Tuple[float, float, float, float, float]: Parameters (alpha1, alpha2, alpha3, alpha4, alpha5) of the Dirichlet distribution.
            """
            return (self.prior_params[0] + self.bucket_02,
                    self.prior_params[1] + self.bucket_24,
                    self.prior_params[2] + self.bucket_46,
                    self.prior_params[3] + self.bucket_68,
                    self.prior_params[4] + self.bucket_810)

    class ResponseFormat(BaseModel):
        """
        Belief about the support for the hypothesis.

        Attributes:
            belief (str): Belief about the support for the hypothesis. Choices are:
                "0-0.2": Hypothesis is definitely false.
                "0.2-0.4": Hypothesis may be false.
                "0.4-0.6": Hypothesis is equally likely to be true or false (e.g., because of relevant but contradictory evidence).
                "0.6-0.8": Hypothesis may be true.
                "0.8-1.0": Hypothesis is definitely true.
                "cannot comment": Not enough information to comment on the hypothesis (e.g., due to lack of domain knowledge or lack of relevant evidence).
        """
        belief: str = Field(..., description="Belief about the hypothesis being true",
                            choices=["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0",
                                     "cannot comment"])

    @staticmethod
    def parse_response(response: List[dict], prior_params: Tuple[float, float, float, float, float] = (
            0.2, 0.2, 0.2, 0.2, 0.2), weight: float = 1.0) -> 'BeliefCategoricalNumeric.DistributionFormat':
        """
        Parse the response from the LLM into a DistributionFormat.

        Args:
            response (dict): The response from the LLM containing belief counts.
            prior_params (Tuple[float, float, float, float, float]): Parameters for the prior Dirichlet distribution.
            weight (float): Weight to apply to the counts (default is 1.0).

        Returns:
            BeliefCategoricalNumeric.DistributionFormat: Parsed distribution format.
        """
        cannot_comment = sum(1 for _res in response if _res["belief"] == "cannot comment")
        bucket_02 = weight * sum(1 for _res in response if _res["belief"] == "0-0.2")
        bucket_24 = weight * sum(1 for _res in response if _res["belief"] == "0.2-0.4")
        bucket_46 = weight * sum(1 for _res in response if _res["belief"] == "0.4-0.6")
        bucket_68 = weight * sum(1 for _res in response if _res["belief"] == "0.6-0.8")
        bucket_810 = weight * sum(1 for _res in response if _res["belief"] == "0.8-1.0")
        n = weight * (len(response) - cannot_comment)  # Exclude responses with "cannot comment"

        return BeliefCategoricalNumeric.DistributionFormat(
            n=n,
            bucket_02=bucket_02,
            bucket_24=bucket_24,
            bucket_46=bucket_46,
            bucket_68=bucket_68,
            bucket_810=bucket_810,
            prior_params=prior_params
        )

    @staticmethod
    def kl_divergence(distr1: 'BeliefCategoricalNumeric.DistributionFormat',
                      distr2: 'BeliefCategoricalNumeric.DistributionFormat') -> float:
        """
        Compute the KL divergence between two distributions.
        Args:
            distr1 (BeliefCategoricalNumeric.DistributionFormat): First distribution.
            distr2 (BeliefCategoricalNumeric.DistributionFormat): Second distribution.
        Returns:
            float: KL divergence between the two distributions.
        """
        alpha = np.array(distr1.get_params())
        beta = np.array(distr2.get_params())

        sum_alpha = np.sum(alpha)
        sum_beta = np.sum(beta)

        term1 = gammaln(sum_alpha) - np.sum(gammaln(alpha))
        term2 = -gammaln(sum_beta) + np.sum(gammaln(beta))
        term3 = np.sum((alpha - beta) * (psi(alpha) - psi(sum_alpha)))

        return term1 + term2 + term3


class BeliefGauss:
    """
    A distribution of beliefs about the hypothesis using Gaussian mean and standard deviation samples.

    Attributes:
        n: Number of samples used to compute the distribution
        mean: Mean probability of the hypothesis being true
        stddev: Standard deviation of the probabilities
        prior_params: Parameters for the prior Gaussian distribution (mean, stddev)
        samples: Dictionary containing means and standard deviations of the samples
        weight: Weight to apply to the counts (default is 1.0)
    """

    class DistributionFormat:
        def __init__(self,
                     n: float = Field(..., description="Number of samples used to compute the distribution"),
                     mean: float = Field(..., description="Mean probability of the hypothesis being true"),
                     stddev: float = Field(..., description="Standard deviation of the probabilities"),
                     prior_params: Tuple[float, float] = (0.5, 5),
                     samples=None,
                     weight=1.0,
                     **kwargs):
            self.n = n
            self.samples = samples
            self.weight = weight
            self._empirical_mean, self._empirical_stddev = fuse_gaussians(self.samples["means"],
                                                                          self.samples["stddevs"])
            self.mean = mean
            self.stddev = stddev
            self.prior_params = prior_params  # Parameters for the prior Gaussian distribution (mean, stddev)

        def __repr__(self):
            return f"BeliefGauss.DistributionFormat(n={self.n}, mean={self.mean}, stddev={self.stddev})"

        def to_dict(self):
            return {
                "_type": "gaussian",
                "prior_params": self.prior_params,
                "samples": self.samples,
                "n": self.n,
                "_empirical_mean": self._empirical_mean,
                "_empirical_stddev": self._empirical_stddev,
                "mean": self.mean,
                "stddev": self.stddev
            }

        def get_mean_belief(self, prior=None, recompute=False) -> float:
            """
            Get the mean of the prior/posterior belief distribution.
            Args:
                prior (BeliefGauss.DistributionFormat): Prior distribution format object.
                recompute (bool): Whether to recompute the mean even if it is already set.
            Returns:
                float: The mean belief probability.
            """
            if recompute:
                # Compute the mean belief using the Gaussian distribution
                self._empirical_mean, self._empirical_stddev = fuse_gaussians(self.samples["means"],
                                                                              self.samples["stddevs"])
                self.mean, self.stddev = fuse_gaussians([self.prior_params[0]] + self.samples["means"],
                                                        [self.prior_params[1]] + self.samples["stddevs"],
                                                        weight=self.weight)

            if prior is not None:
                # Bayesian update
                self.mean, self.stddev = fuse_gaussians(
                    [prior.mean, self.mean], [prior.stddev, self.stddev]
                )

            return self.mean

        def update(self, means: List[float] = None, stddevs: List[float] = None,
                   distr=None, normalize: bool = False):
            """
            Update the distribution with new samples or another distribution.
            """
            if distr is not None:
                self.n += distr.n
                # TOFIX: samples don't take into account the original weight of the distribution
                self.samples["means"].extend(distr.samples["means"])
                self.samples["stddevs"].extend(distr.samples["stddevs"])
                self._empirical_mean, self._empirical_stddev = fuse_gaussians(
                    [self._empirical_mean, distr._empirical_mean], [self._empirical_stddev, distr._empirical_stddev]
                )
                self.mean, self.stddev = fuse_gaussians(
                    [self.mean, distr.mean], [self.stddev, distr.stddev]
                )
            else:
                self.n += len(means)
                self.samples["means"].extend(means)
                self.samples["stddevs"].extend(stddevs)
                self._empirical_mean, self._empirical_stddev = fuse_gaussians([self._empirical_mean] + means,
                                                                              [self._empirical_stddev] + stddevs)
                self.mean, self.stddev = fuse_gaussians([self.mean] + means, [self.stddev] + stddevs)

        def get_params(self) -> Tuple[float, float]:
            """
            Get the parameters of the Gaussian distribution.
            Returns:
                Tuple[float, float]: Parameters (mean, stddev) of the Gaussian distribution.
            """
            return self.mean, self.stddev

    class ResponseFormat(BaseModel):
        """
        Belief (Gaussian) distribution about the support for the hypothesis.

        Attributes:
            belief_mean (float): Mean of the belief distribution (0.0 to 1.0). <0.2: the hypothesis is most likely
            false; 0.2-0.4: the hypothesis may be false; 0.4-0.6: the hypothesis is equally likely to be true or false;
            0.6-0.8: the hypothesis may be true; >0.8: the hypothesis is most likely true.
            belief_stddev (float): Standard deviation of the belief distribution (0.0 to infinity). Smaller values
            indicate more confidence in the belief. If you are not confident in your belief, set the standard deviation
            to a large value (e.g., 1000.0).
        """
        belief_mean: float = Field(..., description="Mean of the belief distribution",
                                   ge=0.0, le=1.0)
        belief_stddev: float = Field(..., description="Standard deviation of the belief distribution",
                                     ge=0.0, le=float('inf'))

    @staticmethod
    def parse_response(response: List[dict], prior_params: Tuple[float, float] = (0.5, 5),
                       weight: float = 1.0) -> 'BeliefGauss.DistributionFormat':
        """
        Parse the response from the LLM into a DistributionFormat.

        Args:
            response (dict): The response from the LLM containing belief probabilities.
            prior_params (Tuple[float, float]): Parameters for the prior Gaussian distribution (mean, stddev).
            weight (float): Weight to apply to the counts (default is 1.0).

        Returns:
            BeliefGauss.DistributionFormat: Parsed distribution format.
        """
        n = weight * len(response)

        means = [_res["belief_mean"] for _res in response]
        stddevs = [_res["belief_stddev"] for _res in response]

        mean, stddev = fuse_gaussians([prior_params[0]] + means, [prior_params[1]] + stddevs, weight=weight)

        return BeliefGauss.DistributionFormat(n=n, mean=mean, stddev=stddev, prior_params=prior_params,
                                              samples={"means": means, "stddevs": stddevs}, weight=weight)

    @staticmethod
    def kl_divergence(distr1: 'BeliefGauss.DistributionFormat',
                      distr2: 'BeliefGauss.DistributionFormat') -> float:
        """
        Compute the KL divergence between two Gaussian distributions.
        Args:
            distr1 (BeliefGauss.DistributionFormat): First distribution.
            distr2 (BeliefGauss.DistributionFormat): Second distribution.
        Returns:
            float: KL divergence between the two distributions.
        """
        mean1, stddev1 = distr1.get_params()
        mean2, stddev2 = distr2.get_params()

        return 0.5 * (np.log(stddev2 ** 2 / stddev1 ** 2) + (stddev1 ** 2 + (mean1 - mean2) ** 2) / stddev2 ** 2 - 1)


class BeliefTrueFalseCat:
    score_per_category = {
        "definitely_false": 0.0,
        "maybe_false": 0.25,
        "uncertain": 0.5,
        "maybe_true": 0.75,
        "definitely_true": 1.0
    }

    class DistributionFormat:
        """
        A distribution of beliefs about the hypothesis using categorical buckets (Categorical).
        Attributes:
            n: Number of samples used to compute the distribution
            definitely_true: Number of "definitely true" responses
            maybe_true: Number of "maybe true" responses
            uncertain: Number of "uncertain" responses
            maybe_false: Number of "maybe false" responses
            definitely_false: Number of "definitely false" responses
            mean: Mean belief probability (optional, computed if not provided)
            prior_params: Parameters for the prior Beta distribution (alpha, beta)
        """

        def __init__(self,
                     n: float = Field(..., description="Number of samples used to compute the distribution"),
                     definitely_true: float = Field(..., description='Number of "definitely true" responses'),
                     maybe_true: float = Field(..., description='Number of "maybe true" responses'),
                     uncertain: float = Field(..., description='Number of "uncertain" responses'),
                     maybe_false: float = Field(..., description='Number of "maybe false" responses'),
                     definitely_false: float = Field(..., description='Number of "definitely false" responses'),
                     mean: float | None = None,
                     prior_params: Tuple[float, float] = (0.5, 0.5),
                     **kwargs):
            self.n = n
            self.definitely_true = definitely_true
            self.maybe_true = maybe_true
            self.uncertain = uncertain
            self.maybe_false = maybe_false
            self.definitely_false = definitely_false
            self.mean = mean
            self._empirical_mean = 0.5
            self.prior_params = prior_params  # Parameters for the prior Beta distribution

        def __repr__(self):
            return (f"BeliefTrueFalseCat.DistributionFormat(n={self.n}, definitely_true={self.definitely_true}, "
                    f"maybe_true={self.maybe_true}, uncertain={self.uncertain}, "
                    f"maybe_false={self.maybe_false}, definitely_false={self.definitely_false})")

        def to_dict(self):
            return {
                "_type": "boolean_cat",
                "prior_params": self.prior_params,
                "n": self.n,
                "definitely_true": self.definitely_true,
                "maybe_true": self.maybe_true,
                "uncertain": self.uncertain,
                "maybe_false": self.maybe_false,
                "definitely_false": self.definitely_false,
                "_empirical_mean": self._empirical_mean,
                "mean": self.mean,
            }

        def get_mean_belief(self, prior=None, recompute=False) -> float:
            """
            Get the mean of the prior/posterior belief distribution.
            Args:
                prior (BeliefTrueFalseCat.DistributionFormat): Prior distribution format object.
                recompute (bool): Whether to recompute the mean even if it is already set.
            Returns:
                float: The mean belief probability.
            """
            if self.mean is None or recompute:
                # Compute the mean belief using the Beta distribution
                alpha1, alpha2 = BeliefTrueFalseCat.get_beta_params_from_cat_samples(
                    self.definitely_true, self.maybe_true, self.uncertain, self.maybe_false, self.definitely_false
                )
                if self.n > 0:
                    self._empirical_mean = alpha1 / self.n
                self.mean = (self.prior_params[0] + alpha1) / (self.n + sum(self.prior_params))

                if prior is not None:
                    # Bayesian update: Beta(n_true + a, n_false + b) where a and b are prior parameters
                    prior_alpha1, prior_alpha2 = BeliefTrueFalseCat.get_beta_params_from_cat_samples(
                        prior.definitely_true, prior.maybe_true, prior.uncertain, prior.maybe_false,
                        prior.definitely_false
                    )
                    post_alpha = prior_alpha1 + prior.prior_params[0]
                    # post_beta = prior_alpha2 + prior.prior_params[1]
                    self.mean = (alpha1 + post_alpha) / (self.n + prior.n + sum(prior.prior_params))
            return self.mean

        def update(self,
                   definitely_true: int = 0,
                   maybe_true: int = 0,
                   uncertain: int = 0,
                   maybe_false: int = 0,
                   definitely_false: int = 0,
                   distr=None,
                   normalize: bool = False):
            """
            Update the distribution with new counts.
            """
            if distr is not None:
                self.definitely_true += distr.definitely_true
                self.maybe_true += distr.maybe_true
                self.uncertain += distr.uncertain
                self.maybe_false += distr.maybe_false
                self.definitely_false += distr.definitely_false
            else:
                self.definitely_true += definitely_true
                self.maybe_true += maybe_true
                self.uncertain += uncertain
                self.maybe_false += maybe_false
                self.definitely_false += definitely_false
            n = distr.n if distr is not None else (
                    definitely_true + maybe_true + uncertain + maybe_false + definitely_false
            )
            if normalize:
                total = self.n + n
                self.definitely_true /= (total / self.n)
                self.maybe_true /= (total / self.n)
                self.uncertain /= (total / self.n)
                self.maybe_false /= (total / self.n)
                self.definitely_false /= (total / self.n)
            else:
                self.n += n
            # Reset mean
            _ = self.get_mean_belief(recompute=True)

        def get_params(self) -> Tuple[float, float]:
            """
            Get the parameters of the Beta distribution.
            Returns:
                Tuple[float, float]: Parameters (alpha, beta) of the Beta distribution.
            """
            alpha1, alpha2 = BeliefTrueFalseCat.get_beta_params_from_cat_samples(
                self.definitely_true, self.maybe_true, self.uncertain, self.maybe_false, self.definitely_false
            )
            return self.prior_params[0] + alpha1, self.prior_params[1] + alpha2

    class ResponseFormat(BaseModel):
        """
        Belief about the support for the hypothesis.

        Attributes:
            belief (str): Belief about the support for the hypothesis. Choices are:
                "definitely true": Hypothesis is definitely true.
                "maybe true": Hypothesis may be true.
                "uncertain": Hypothesis is equally likely to be true or false (e.g., because of relevant but contradictory evidence).
                "maybe false": Hypothesis may be false.
                "definitely false": Hypothesis is definitely false.
                "cannot comment": Not enough information to comment on the hypothesis (e.g., due to lack of domain knowledge or lack of relevant evidence).
        """
        belief: str = Field(..., description="Belief about the hypothesis",
                            choices=["definitely true", "maybe true", "uncertain",
                                     "maybe false", "definitely false", "cannot comment"])

    @staticmethod
    def parse_response(response: List[dict],
                       prior_params: Tuple[float, float] = (0.5, 0.5),
                       weight: float = 1.0) -> 'BeliefTrueFalseCat.DistributionFormat':
        """
        Parse the response from the LLM into a DistributionFormat.

        Args:
            response (dict): The response from the LLM containing belief counts.
            prior_params (Tuple[float, float]): Parameters for the prior Beta distribution (alpha, beta).
            weight (float): Weight to apply to the counts (default is 1.0).

        Returns:
            BeliefTrueFalseCat.DistributionFormat: Parsed distribution format.
        """
        cannot_comment = sum(1 for _res in response if _res["belief"] == "cannot comment")
        definitely_true = weight * sum(1 for _res in response if _res["belief"] == "definitely true")
        maybe_true = weight * sum(1 for _res in response if _res["belief"] == "maybe true")
        uncertain = weight * sum(1 for _res in response if _res["belief"] == "uncertain")
        maybe_false = weight * sum(1 for _res in response if _res["belief"] == "maybe false")
        definitely_false = weight * sum(1 for _res in response if _res["belief"] == "definitely false")
        n = weight * (len(response) - cannot_comment)  # Exclude responses with "cannot comment"

        return BeliefTrueFalseCat.DistributionFormat(
            n=n,
            definitely_true=definitely_true,
            maybe_true=maybe_true,
            uncertain=uncertain,
            maybe_false=maybe_false,
            definitely_false=definitely_false,
            prior_params=prior_params
        )

    @staticmethod
    def kl_divergence(distr1: 'BeliefTrueFalseCat.DistributionFormat',
                      distr2: 'BeliefTrueFalseCat.DistributionFormat') -> float:
        """
        Compute the KL divergence between two distributions.
        Args:
            distr1 (BeliefTrueFalseCat.DistributionFormat): First distribution.
            distr2 (BeliefTrueFalseCat.DistributionFormat): Second distribution.
        Returns:
            float: KL divergence between the two distributions.
        """
        alpha1, beta1 = distr1.get_params()
        alpha2, beta2 = distr2.get_params()
        term1 = betaln(alpha2, beta2) - betaln(alpha1, beta1)
        term2 = (alpha1 - alpha2) * psi(alpha1)
        term3 = (beta1 - beta2) * psi(beta1)
        term4 = (alpha2 - alpha1 + beta2 - beta1) * psi(alpha1 + beta1)
        return term1 + term2 + term3 + term4

    @staticmethod
    def get_beta_params_from_cat_samples(definitely_true: float, maybe_true: float, uncertain: float,
                                         maybe_false: float, definitely_false: float) -> Tuple[float, float]:
        """
        Convert categorical counts into parameters for a Beta distribution.

        Args:
            definitely_true: Count of "definitely true" responses.
            maybe_true: Count of "maybe true" responses.
            uncertain: Count of "uncertain" responses.
            maybe_false: Count of "maybe false" responses.
            definitely_false: Count of "definitely false" responses.

        Returns:
            Tuple[float, float]: Parameters (alpha, beta) for the Beta distribution.
        """
        total = definitely_true + maybe_true + uncertain + maybe_false + definitely_false
        alpha = definitely_true * BeliefTrueFalseCat.score_per_category["definitely_true"] + \
                maybe_true * BeliefTrueFalseCat.score_per_category["maybe_true"] + \
                uncertain * BeliefTrueFalseCat.score_per_category["uncertain"] + \
                maybe_false * BeliefTrueFalseCat.score_per_category["maybe_false"] + \
                definitely_false * BeliefTrueFalseCat.score_per_category["definitely_false"]
        beta = total - alpha
        return alpha, beta


BELIEF_MODE_TO_CLS = {
    "boolean": BeliefTrueFalse,
    "boolean_cat": BeliefTrueFalseCat,
    "categorical": BeliefCategorical,
    "categorical_numeric": BeliefCategoricalNumeric,
    "gaussian": BeliefGauss
}


def get_belief(
        hypothesis: str,
        evidence: Optional[List[Dict[str, str]]] = None,
        model: str = "gpt-4o",
        belief_mode: str = "boolean",
        n_samples: int = 5,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        use_llm_prior: bool = False,
        explicit_prior=None,
        n_retries=3,
        weight: float = 1.0
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
        use_llm_prior: Whether to use implicit Bayesian posterior
        explicit_prior: Optional prior distribution to use for Bayesian updates
        n_retries: Number of retries for querying the LLM in case of errors
        weight: Weight to apply to the empirical distribution
    """
    belief_cls = BELIEF_MODE_TO_CLS.get(belief_mode)
    if belief_cls is None:
        raise ValueError(f"Unknown belief_mode '{belief_mode}'; expected one of {list(BELIEF_MODE_TO_CLS.keys())}")

    # Construct the system prompt based on whether we are eliciting prior, implicit posterior, or explicit posterior beliefs
    _system_msgs = [
        "You are a research scientist skilled at analyzing scientific hypotheses. Your task is to provide your belief about the given hypothesis."
    ]
    if evidence is not None:
        # posterior belief
        _system_msgs.append(
            "Use the provided evidence collected from running experiments to help make your decision. Carefully consider each piece of evidence and decide whether and how any of them affects your belief about the current hypothesis. Note that evidence from previous studies may have an indirect bearing on the hypothesis, so think about how they might relate to the hypothesis even if they do not directly test it."
        )
    # else:  # prior belief
    if use_llm_prior:
        # implicit posterior
        _system_msgs.append(
            "Use your prior knowledge of the research domain to help in your assessment of the hypothesis."
        )
    else:
        # explicit posterior
        assert evidence is not None
        _system_msgs.append(
            "Disregard any prior beliefs you have about the hypothesis and focus only on the provided evidence."
        )
    system_prompt = {
        "role": "system",
        "content": " ".join(_system_msgs)
    }

    hypothesis_msg = {
        "role": "user",
        "content": f"Hypothesis: {hypothesis}\n\nCarefully reason before making your assessment."
    }

    all_msgs = [system_prompt]
    if evidence is not None:
        all_msgs += evidence
    all_msgs.append(hypothesis_msg)

    distribution, mean_belief = None, None
    for attempt in range(n_retries):
        try:
            response = query_llm(all_msgs, model=model, n_samples=n_samples,
                                 temperature=temperature, reasoning_effort=reasoning_effort,
                                 response_format=belief_cls.ResponseFormat)
            prior_params_or_none = {}
            if explicit_prior is not None:
                prior_params_or_none["prior_params"] = explicit_prior.get_params()
            distribution = belief_cls.parse_response(response, weight=weight, **prior_params_or_none)
            # Compute and store the mean belief
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
                                          belief_mode="boolean", evidence_msg=None, prior=None, evidence_weight=1.0):
    """
    Calculate prior and posterior belief distributions for a hypothesis.

    Args:
        node: MCTSNode instance containing node information and messages or a dictionary with node data
        n_samples: Number of samples to draw from the LLM
        model: The LLM model to use for querying
        temperature: Temperature for sampling
        reasoning_effort: Reasoning effort for o-series models
        implicit_bayes_posterior: Whether to use implicit Bayesian posterior
        surprisal_width: Minimum difference in mean prior and posterior probabilities required to count as a surprisal
        belief_mode: The belief mode to use for parsing responses (e.g., "boolean", "categorical")
        evidence_msg: Optional evidence messages to condition the posterior belief
        prior: Optional pre-computed prior distribution to use for posterior calculation
        evidence_weight: Weight to apply to the evidence when calculating the posterior belief
    """

    # MODEL_CTXT_LIMITS = {
    #     "o4-mini": 200_000,
    #     "gpt-4o": 128_000,
    # }
    belief_cls = BELIEF_MODE_TO_CLS.get(belief_mode)
    if belief_cls is None:
        raise ValueError(f"Unknown belief_mode '{belief_mode}'; expected one of {list(BELIEF_MODE_TO_CLS.keys())}")

    if type(node) is MCTSNode:
        hypothesis = node.hypothesis
        query = node.query  # Contains the hypothesis and experiment plan
        code_output = node.code_output
        analysis = node.analysis
        review = node.review
    else:
        hypothesis = node["hypothesis"]
        query = node.get("query", "N/A")
        code_output = node.get("code_output", "N/A")
        analysis = node.get("analysis", "N/A")
        review = node.get("review", "N/A")

    if hypothesis is None:
        return None, None, None, None

    if evidence_msg is None:
        evidence_msg = [{
            "role": "user",
            "content": get_context_string(query, code_output, analysis, review, include_code_output=True)
        }]

    if prior is None:
        prior, _ = get_belief(
            hypothesis=hypothesis,
            evidence=None,
            model=model,
            belief_mode=belief_mode,
            n_samples=n_samples,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            use_llm_prior=True,
        )

    posterior, _ = get_belief(
        hypothesis=hypothesis,
        evidence=evidence_msg,
        model=model,
        belief_mode=belief_mode,
        n_samples=n_samples,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        use_llm_prior=implicit_bayes_posterior,
        explicit_prior=prior,
        weight=evidence_weight
    )

    if prior is None or posterior is None:
        raise ValueError("Belief distribution could not be computed.")

    belief_change = abs(posterior.mean - prior.mean)
    kl_divergence = belief_cls.kl_divergence(posterior, prior)

    return prior, posterior, belief_change, kl_divergence


if __name__ == "__main__":

    # Unit test
    from mcts_utils import load_mcts_from_json

    path = "/Users/dagarwal/code/github/allenai/autods-humans/debug/20250826-181702"
    root, nodes_by_level = load_mcts_from_json(path)
    belief_kl = []
    prior_posterior = []
    for level, nodes in nodes_by_level.items():
        for node in nodes:
            if node.prior is not None:
                belief_cls = BELIEF_MODE_TO_CLS[node.prior.to_dict()["_type"]]
                prior = node.prior
                posterior = node.posterior
                belief_change = round(posterior.mean - prior.mean, 2)
                kl_div = round(belief_cls.kl_divergence(posterior, prior), 2)
                belief_kl.append((belief_change, kl_div))
                prior_posterior.append((prior.get_params(), posterior.get_params()))
    print(f"Total nodes: {len(belief_kl)}\n\n")
    # Print statistics (percentiles, mean, std) of belief change and KL divergence
    belief_changes = [abs(bc[0]) for bc in belief_kl]
    kl_divergences = [bc[1] for bc in belief_kl]
    print(f"Belief Change - Mean: {np.mean(belief_changes):.2f}, Std: {np.std(belief_changes):.2f}")
    print(f"Belief Change - Min: {np.min(belief_changes):.2f}, Max: {np.max(belief_changes):.2f}")
    print(f"Belief Change - 25th Percentile: {np.percentile(belief_changes, 25):.2f}")
    print(f"Belief Change - 50th Percentile: {np.percentile(belief_changes, 50):.2f}")
    print(f"Belief Change - 75th Percentile: {np.percentile(belief_changes, 75):.2f}")

    print(f"KL Divergence - Mean: {np.mean(kl_divergences):.2f}, Std: {np.std(kl_divergences):.2f}")
    print(f"KL Divergence - Min: {np.min(kl_divergences):.2f}, Max: {np.max(kl_divergences):.2f}")
    print(f"KL Divergence - 25th Percentile: {np.percentile(kl_divergences, 25):.2f}")
    print(f"KL Divergence - 50th Percentile: {np.percentile(kl_divergences, 50):.2f}")
    print(f"KL Divergence - 75th Percentile: {np.percentile(kl_divergences, 75):.2f}\n\n")

    # Print a table of belief change, KL divergence, and prior/posterior parameters in sorted order of KL divergence
    print(f"{'Belief Change':<20} {'KL Divergence':<20} {'Prior Params':<50} {'Posterior Params':<50}")
    sorted_tuples = sorted(zip(belief_kl, prior_posterior), key=lambda x: x[0][1])
    for (belief_change, kl_div), (prior_params, posterior_params) in sorted_tuples:
        prior_params_str = ", ".join(f"{p:.2f}" for p in prior_params)
        posterior_params_str = ", ".join(f"{p:.2f}" for p in posterior_params)
        print(
            f"{belief_change:<20} {round(kl_div / np.mean(kl_divergences), 2):<20} {prior_params_str:<50} {posterior_params_str:<50}")

    print("\n\n")

    sorted_tuples = sorted(zip(belief_kl, prior_posterior), key=lambda x: abs(x[0][0]))
    for (belief_change, kl_div), (prior_params, posterior_params) in sorted_tuples:
        prior_params_str = ", ".join(f"{p:.2f}" for p in prior_params)
        posterior_params_str = ", ".join(f"{p:.2f}" for p in posterior_params)
        print(
            f"{belief_change:<20} {round(kl_div / np.mean(kl_divergences), 2):<20} {prior_params_str:<50} {posterior_params_str:<50}")

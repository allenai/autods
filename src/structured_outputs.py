from pydantic import BaseModel, model_validator
from typing import Optional
from typing_extensions import Self


class Relationship(BaseModel):
    """
    Represents a relationship between two variables in a hypothesis.

    Attributes:
        explanatory (str): The independent/explanatory variable in the relationship
        response (str): The dependent/response variable in the relationship
        relationship (str): Description of how the explanatory variable affects the response variable
    """
    explanatory: str
    response: str
    relationship: str


class HypothesisDimensions(BaseModel):
    """
    Structured representation of the key dimensions of a hypothesis.

    Attributes:
        contexts (list[str]): List of boundary conditions and assumptions under which the hypothesis holds
        variables (list[str]): List of key concepts/variables involved in the hypothesis
        relationships (list[Relationship]): List of causal relationships between pairs of variables
    """
    contexts: list[str]
    variables: list[str]
    relationships: list[Relationship]


class Hypothesis(BaseModel):
    """
    A declarative sentence about the state of the world whose truth value may be inferred from the given dataset(s) using an experiment.

    Attributes:
        hypothesis (str): The hypothesis statement
        dimensions (HypothesisDimensions): Structured dimensions of the hypothesis
    """
    hypothesis: str
    dimensions: HypothesisDimensions


class ExperimentPlan(BaseModel):
    """
    Represents the experiment plan with a title, objective, steps, and deliverables.

    Attributes:
        objective (str): The main goal or objective of the experiment
        steps (str): List of steps to be followed to implement the experiment
        deliverables (str): List of expected outcomes or deliverables from the experiment
    """
    objective: str
    steps: str
    deliverables: str


class Experiment(BaseModel):
    """
        Represents an experiment with a hypothesis and corresponding experiment plan.
        Attributes:
            hypothesis (str): A natural-language hypothesis representing an assertion about the world
            experiment_plan (ExperimentPlan): The structured experiment plan to verify the hypothesis
        """
    hypothesis: str
    experiment_plan: ExperimentPlan

class ExperimentHypothesis(BaseModel):
    """
        Represents an experiment with an experiment plan and a hypothesis.
        Attributes:
            experiment_plan (ExperimentPlan): A structured experiment plan to verify a hypothesis
            hypothesis (str): A natural-language hypothesis representing an assertion about the world that can be
                              tested by the experiment
        """
    experiment_plan: ExperimentPlan
    hypothesis: str


class ExperimentList(BaseModel):
    """
    A collection of experiments.

    Attributes:
        experiments (list[Experiment]): List of Experiment objects
    """
    experiments: list[Experiment]

class ExperimentHypothesisList(BaseModel):
    """
    A collection of experiment hypotheses.

    Attributes:
        experiments (list[ExperimentHypothesis]): List of ExperimentHypothesis objects
    """
    experiments: list[ExperimentHypothesis]


class ExperimentCode(BaseModel):
    """
    Contains the code implementation for an experiment.

    Attributes:
        code (str): The actual code to be executed for the experiment
    """
    code: str


class ProgramCritique(BaseModel):
    """
    Feedback on experiment code implementation.

    Attributes:
        fixes (list[str]): List of suggested fixes or improvements for the code
    """
    fixes: list[str]


class ExperimentAnalyst(BaseModel):
    """
    Analysis of experiment results.

    Attributes:
        success (bool): Whether the experiment was successful
        analysis (Optional[str]): Detailed analysis of the experiment outcomes
    """
    success: bool
    analysis: str

    @model_validator(mode='after')
    def analysis_required_on_success(self) -> Self:
        if self.success and self.analysis is None:
            raise ValueError('analysis is required when success is True')
        return self


class ExperimentReviewer(BaseModel):
    """
    Review of an experiment's execution and results.

    Attributes:
        success (bool): Whether the experiment was successful
        feedback (str | None): Required feedback when experiment fails, optional otherwise

    Raises:
        ValueError: If success is False and no feedback is provided
    """
    success: bool
    feedback: str

    @model_validator(mode='after')
    def feedback_required_on_failure(self) -> Self:
        if not self.success and self.feedback is None:
            raise ValueError('feedback is required when success is False')
        return self


class ImageAnalysis(BaseModel):
    """
    Structured representation of plot axes and related analysis information.

    Attributes:
        title (str): The title of the plot
        x_axis_label (str): Label for the x-axis
        y_axis_label (str): Label for the y-axis
        x_axis_range (list[int | float]): Range of values on the x-axis
        y_axis_range (list[int | float]): Range of values on the y-axis
        data_trends (list[str]): List of observed trends in the data
        statistical_insights (list[str]): List of statistical observations and metrics
        annotations_and_legends (list[str]): List of plot annotations and legend descriptions
    """
    plot_type: str
    title: str
    x_axis_label: str
    y_axis_label: str
    x_axis_range: list[int] | list[float]
    y_axis_range: list[int] | list[float]
    data_trends: list[str]
    statistical_insights: list[str]
    annotations_and_legends: list[str]


class ExecutionResult(BaseModel):
    exit_code: int
    result: str

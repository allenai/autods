from autogen import ConversableAgent, UserProxyAgent
from src.structured_outputs import ExperimentList, ExperimentCode, ExperimentAnalyst, ExperimentReviewer, Experiment, \
    ExperimentHypothesisList
import os
import json
from autogen.coding import LocalCommandLineCodeExecutor
from typing import Tuple

import copy
from typing import List, Dict
import autogen.agentchat.contrib.capabilities.transforms as transforms
from autogen.agentchat.contrib.capabilities import transform_messages

IMAGE_ANALYSIS_PATCH = """\
import matplotlib.pyplot as plt
import functools
from io import BytesIO
import base64
from openai import OpenAI


client = OpenAI()

image_analyst_prompt = '''Please analyze the given plot image and provide the following:

1. Plot Type: Identify the type of plot (e.g., heatmap, bar plot, scatter plot) and its purpose.
2. Axes:
    * Titles and labels, including units.
    * Value ranges for both axes.
3. Data Trends:
    * For scatter plots: note trends, clusters, or outliers.
    * For bar plots: highlight the tallest and shortest bars and patterns.
    * For heatmaps: identify areas of high and low values.
    etc...
4. Annotations and Legends: Describe key annotations or legends.
5. Statistical Insights: Provide insights based on the information presented in the plot.'''


def image_to_text():
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)  # Get the current figure
        with BytesIO() as buf:
            # Save the figure to a PNG buffer
            fig.savefig(buf, format='png', dpi=200)
            buf.seek(0)
            # Encode image to base64
            base64_image = base64.b64encode(buf.read()).decode('utf-8')
            messages = [
                {
                    'role': 'system',
                    'content': 'You are a research scientist responsible for analyzing plots and figures from running experiments and providing detailed descriptions.'
                },
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': image_analyst_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
            # Get image analysis from the LLM
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=1000,
            )
            analysis = response.choices[0].message.content
            print(f"\\n=== Plot Analysis (fig. {fig_num}) ===\\n")
            print(analysis)
            print("\\n" + "="*50)

        plt.close(fig)


def patch_matplotlib_show():
    # Replace plt.show with our custom function
    plt.show = functools.partial(image_to_text)


# Apply the patch
patch_matplotlib_show()
"""


class CodeBlockWrapperTransform(transforms.MessageTransform):

    def apply_transform(self, messages: List[Dict]) -> List[Dict]:
        # Deep copy messages to avoid modifying the original
        transformed_messages = copy.deepcopy(messages)
        message = transformed_messages[-1]

        try:
            code = json.loads(message["content"]).get("code", "# Failed to parse code from message")
        except json.JSONDecodeError:
            code = "# Failed to parse code from message"

        message["content"] = f"```python\n{IMAGE_ANALYSIS_PATCH}\n\n{code}\n```"

        return transformed_messages

    def get_logs(self, pre_transform_messages: List[Dict], post_transform_messages: List[Dict]) -> Tuple[str, bool]:
        return "CodeBlockWrapperTransform", True


def get_openai_config(api_key: str | None = None, temperature: float | None = None,
                      reasoning_effort: str | None = None, timeout: int = 600, model_name="o4-mini"):
    config = {
        "api_type": "openai",
        "model": model_name,
        "timeout": timeout,
        "api_key": api_key,
        "max_retries": 3,
        "cache_seed": None  # Disabling caching also addresses this bug: https://github.com/ag2ai/ag2/issues/1103
    }
    if temperature is not None:
        config["temperature"] = temperature

    # Make o-series specific changes
    if model_name.startswith("o"):
        if reasoning_effort is not None:
            config["reasoning_effort"] = reasoning_effort  # Defaults to medium
    else:
        config["logprobs"] = True

    return config


def get_agents(work_dir, model_name="o4-mini", temperature=None, reasoning_effort=None, branching_factor=3,
               user_query=None, experiment_first=False, code_timeout=30 * 60) -> dict[str, ConversableAgent]:
    llm_config = get_openai_config(api_key=os.getenv("OPENAI_API_KEY"), model_name=model_name, temperature=temperature,
                                   reasoning_effort=reasoning_effort)

    # Create token limit transform
    token_limit_capability = transform_messages.TransformMessages(transforms=[
        transforms.MessageTokenLimiter(max_tokens_per_message=10_000, min_tokens=12_000)
    ])

    # Experiment Generator
    _user_query_or_empty = f"{user_query}\n\n" if user_query is not None else ""

    experiment_generator = ConversableAgent(
        name="experiment_generator",
        llm_config={**llm_config,
                    "response_format": ExperimentList if not experiment_first else ExperimentHypothesisList},
        system_message=(
            'You are a research scientist who is interested in doing open-ended, data-driven research using the provided dataset(s). '
            f'{_user_query_or_empty}'
            f'Be creative and think of new and interesting verifiable {"experiments" if experiment_first else "hypotheses"} and corresponding {"hypotheses" if experiment_first else "experiments"}. '
            'The hypothesis should be a falsifiable statement that can be sufficiently tested by an experiment using the provided data. '
            'Explain in natural language what this experiment plan is so that a programmer can implement it (do not provide the code yourself). '
            'Remember, you are interested in open-ended research, so your proposals may be exploratory in nature and may have only an indirect connection to the previous explorations provided. '
            'Here are some instructions that you must follow:\n'
            '1. Strictly use only the dataset(s) provided and do not simulate dummy/synthetic data or columns that cannot be derived from the existing columns.\n'
            '2. Each hypothesis (and experiment plan) should be creative, independent, and self-contained.\n'
            '3. Use the prior experiments/hypotheses as inspiration to think of interesting and creative new experiments/hypotheses. However, do not repeat the same experiments/hypotheses.\n\n'
            'Here is a possible approach to coming up with a new hypothesis and experiment plan:\n'
            '1. Find an interesting context: this could be a specific subset of the data. E.g., if the dataset has multiple categorical variables, you could split the data based on specific values of such variables, which would then allow you to validate a hypothesis in the specific contexts defined by the values of those variables.\n'
            '2. Find interesting variables: these could be the columns in the dataset that you find interesting or relevant to the context. You are allowed and encouraged to create composite variables derived from the existing variables.\n'
            '3. Find interesting relationships: these are interactions between the variables that you find interesting or relevant to the context. You are encouraged to propose experiments involving complex predictive or causal models.\n'
            '4. You must require that your proposed hypotheses are verifiable using robust statistical tests. Remember, your programmer can install python packages via pip which can allow it to write code for complex statistical analyses.\n'
            '5. Multiple datasets: If you are provided with more than one dataset, then try to also propose hypotheses that utilize contexts, variables, and relationships across datasets, e.g., this may involve using join or similar operations.\n\n'
            'Generally, in typical data-driven research, you will need to explore and visualize the data for possible high-level insights, clean, transform, or derive new variables from the dataset to be suited for the investigation, deep-dive into specific parts of the data for fine-grained analysis, perform data modeling, and run statistical tests. '
            f'Now, generate exactly {branching_factor} new hypotheses with their experiment plans.'
        ),
        human_input_mode="NEVER",
    )

    install_snippet = ("""\nimport subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])\n\n\n""")

    # Experiment Programmer
    experiment_programmer = ConversableAgent(
        name="experiment_programmer",
        llm_config={**llm_config, "response_format": ExperimentCode},
        system_message=(
            'You are a scientific experiment programmer proficient in writing python code given an experiment plan. '
            'Your code will be included in a python file that is executed and any relevant results should be printed to standard out or presented using plt.show appropriately. '
            'Make sure you provide python code in the proper format to execute. '
            'Ensure your code is clean and concise, and include debug statements only when they are absolutely necessary. '
            'Use only the dataset given and do not assume any other files are available. The state is not preserved between code blocks, so do not assume any variables or imports from previous code blocks. '
            'Import any libraries you need to use. Always attempt to import a library before installing it (it may already be installed). '
            'If you need to install a library, use the following code example:'
            f'{install_snippet}'
            'When installing python packages, use the --quiet option to minimize unnecessary output.'
            'Prefer using installed libraries over installing new libraries whenever possible. '
            'If possible, instead of downgrading library versions, try to adapt your code to work with a more updated version that is already installed. '
            'Never attempt to create a new environment. Always use the current environment. '
            'If the code requires generating plots, use plt.show (not plt.savefig).  '
            'Avoid printing the whole data structure to the console directly if it is large; instead, print concise results that are directly relevant to the experiment. '
            'You are allowed 6 total attempts to run the code, including debugging attempts.\n\n'
            'Debugging instructions:\n'
            '1. Only debug if you are either unsure about the executability or validity of the code (i.e., whether it satisfies the proposed experiment).\n'
            '2. If the code you are writing is intended for debugging, the first line of your code must be "# [debug]" only.\n'
            '3. DO NOT use "[debug]" anywhere else in your code.\n'
            '4. DO NOT combine any debug code and the actual experiment implementation code; keep them separate.\n'
            '5. For each experiment, you are allowed to debug at most 3 times.\n'
            '6. As much as possible, minimize the number of debugging steps you use.'
        ),
        human_input_mode="NEVER",
    )

    # Experiment Analyst
    experiment_analyst = ConversableAgent(
        name="experiment_code_analyst",
        llm_config={**llm_config, "response_format": ExperimentAnalyst},
        system_message=(
            'You are a research scientist responsible for evaluating the code execution output for a scientific experiment written by a programmer. '
            'If no code was executed, there was an error, or the code fails silently, return the success status as **false**. '
            'If the code includes a line "# [debug]" i.e "[debug]" as a comment, strictly treat this as a debugging experiment. '
            'In such cases, strictly return the success status as **false**, provide information that it was a debug code execution, '
            'give feedback and request the experiment to be retried with the new information. '
            'Otherwise, analyze the results and provide a short summary of the code output.'
        ),
        human_input_mode="NEVER",
    )

    # Experiment Reviewer
    experiment_reviewer = ConversableAgent(
        name="experiment_reviewer",
        llm_config={**llm_config, "response_format": ExperimentReviewer},
        system_message=(
            'You are a research scientist responsible for holistically reviewing the entire experiment pipeline, i.e., the generated code, the output, and the analysis w.r.t. the original experiment plan. '
            'Assess whether the experiment was faithfully implemented, i.e., whether the implementation follows the experiment plan without significant deviation and whether the hypothesis was in fact tested sufficiently. '
            'If you find issues or inconsistencies in any part of the experiment pipeline, return the success status as **false** and provide feedback about what is wrong. '
            'Otherwise, return the success status as **true** and provide a summary of the hypothesis, experiment results, and findings.'
        ),
        human_input_mode="NEVER",
    )

    # Experiment Reviser
    experiment_reviser = ConversableAgent(
        name="experiment_reviser",
        llm_config={**llm_config, "response_format": Experiment},
        system_message=(
            'You are a research scientist revisiting the most recent experiment, which could not be conducted correctly due to issues in the code or the formulation of the experiment plan,'
            'as indicated by the reviewer. Your goal is to revise this failed experiment plan by addressing the issues and limitations pointed out by the reviewer. '
            'The revised experiment plan should still aim to validate the most recent hypothesis. '
            'Do not provide the code yourself but explain in natural language what the experiment should do for a programmer. '
            'Strictly use only the dataset provided and do not create synthetic data or columns that cannot be derived from the given columns. '
            'The experiment should be creative, independent, and self-contained. '
            'Generally, in typical data-driven research, you will need to explore and visualize the data for possible high-level insights, clean, transform, or derive new variables from the dataset to be suited for the investigation, deep-dive into specific parts of the data for fine-grained analysis, perform data modeling, and run statistical tests.'
        ),
        human_input_mode="NEVER",
    )

    ## Timeout Code Executor
    executor = LocalCommandLineCodeExecutor(
        timeout=code_timeout,  # Timeout in seconds
        work_dir=work_dir,
        # virtual_env_context=create_virtual_env(os.path.join(work_dir, ".venv"))  # TODO: Fix virtual env creation
    )
    # TODO: Fix docker-based execution
    # executor = DockerCommandLineCodeExecutor(
    #     # image="python:3.11-alpine",
    #     timeout=30 * 60,  # Timeout in seconds
    #     work_dir=work_dir,
    #     # virtual_env_context=create_virtual_env(os.path.join(work_dir, ".venv"))
    # )

    # Create an agent with code executor configuration.
    code_executor = ConversableAgent(
        "code_executor",
        llm_config=False,
        code_execution_config={"executor": executor},
        human_input_mode="NEVER",
    )
    # Apply image analysis patch to the code executor
    transform_messages_capability = transform_messages.TransformMessages(transforms=[CodeBlockWrapperTransform()])
    transform_messages_capability.add_to_agent(code_executor)

    user_proxy = UserProxyAgent(
        name="user_proxy",
        description="Responsible for providing the initial query",
        code_execution_config=False,
        human_input_mode="NEVER"
    )

    agents = [experiment_generator, experiment_programmer, experiment_analyst, experiment_reviewer, experiment_reviser,
              code_executor, user_proxy]

    # Apply token limit to all agents
    for agent in agents:
        token_limit_capability.add_to_agent(agent)

    agents_dict = {agent.name: agent for agent in agents}
    return agents_dict

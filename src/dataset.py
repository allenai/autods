import json
import os

from src.utils import fetch_from_s3


def load_blade_metadata(info_path: str) -> dict:
    with open(info_path, 'r') as file:
        return json.load(file)


def load_info_metadata(info_path: str) -> dict:
    with open(info_path, 'r') as file:
        return json.load(file)


def get_blade_description(info_path: str) -> str:
    """
    Generate a human-readable description of the dataset based on its metadata.

    Args:
        info_path: Path to info.json file

    Returns:
        str: Formatted description of the dataset
    """
    metadata = load_blade_metadata(info_path)
    data = metadata.get('data_desc', {})

    description = []

    # Add header
    description.append("Dataset Description")

    # Add dataset section
    description.append("\nDatasets:")

    # Extract dataset name from second-to-last folder in data_path
    data_path = metadata.get("data_path", "")
    dataset_name = os.path.basename(os.path.dirname(data_path)) or "Unnamed Dataset"
    description.append(f"Dataset Name: {dataset_name}")

    dataset_desc = data.get("dataset_description", "No description available.")
    description.append(f"Dataset Description: {dataset_desc}")

    # Add columns
    description.append("\nColumns:")
    fields = data.get("fields", [])
    for field in fields:
        col_name = field.get("column", "Unnamed")
        col_desc = field.get("properties", {}).get("description", "No description available.")
        description.append(f"\n{col_name}:")
        description.append(f"  {col_desc}")

    return "\n".join(description)


def load_ai2_metadata(info_path: str) -> dict:
    with open(info_path, 'r') as file:
        return json.load(file)


def get_ai2_description(ai2_metadata_path: str) -> str:
    """
    Generate a human-readable description of the AI2 dataset based on its metadata.

    Args:
        ai2_metadata_path: Path to the AI2-style metadata JSON file

    Returns:
        str: Formatted description of the dataset
    """
    metadata = load_ai2_metadata(ai2_metadata_path)

    description = []

    # Add header
    description.append("Dataset Description")

    # Add dataset section
    description.append("\nDatasets:")
    for dataset in metadata.get("datasets", []):
        name = dataset.get("name", "Unnamed Dataset")
        desc = dataset.get("description", "No description available.")
        description.append(f"Dataset Name: {name}")
        description.append(f"Dataset Description: {desc}")

        # Add columns
        description.append("\nColumns:")
        for col in dataset.get("columns", {}).get("raw", []):
            col_name = col.get("name", "Unnamed")
            col_desc = col.get("description", "No description available.")
            description.append(f"\n{col_name}:")
            description.append(f"  {col_desc}")

    return "\n".join(description)


def load_dataset_metadata(dataset_metadata_path: str, dataset_metadata_key: str = None) -> dict:
    with open(dataset_metadata_path, 'r') as file:
        dataset_metadata = json.load(file)
    if dataset_metadata_key is not None:
        dataset_metadata = dataset_metadata[dataset_metadata_key]
    return dataset_metadata


def get_dataset_description(dataset_metadata_path: str) -> str:
    """
    Generate a human-readable description of the dataset based on its metadata.
    
    Args:
        dataset_metadata_path: Path to the dataset metadata JSON file
    
    Returns:
        str: Formatted description of the dataset
    """

    metadata = load_dataset_metadata(dataset_metadata_path)
    description = []

    # Add header
    description.append("##### DATASET DESCRIPTION #####")
    # Add dataset info
    description.append("\n### DATASETS: ###\n")
    for dataset in metadata['datasets']:
        description.append(f"Dataset Name: {dataset['name']}")
        description.append(f"Dataset Description: {dataset['description']}")
        description.append("\n### COLUMNS: ###")
        for col in dataset['columns']['raw']:
            description.append(f"\n{col['name']}:")
            description.append(f"  {col['description']}")

    return "\n".join(description)


def get_datasets_fpaths(dataset_metadata: str, is_blade=False) -> (list, str):
    is_s3 = dataset_metadata.startswith("s3")
    _dataset_metadata = dataset_metadata
    if is_s3:
        # Download the metadata and get the local path
        _dataset_metadata = fetch_from_s3([dataset_metadata])[0]
    # Read the json, loop through "datasets" key, then extract dataset path from "name" key
    with open(_dataset_metadata, 'r') as file:
        obj = json.load(file)
    # Get the dataset paths
    datasets = []
    if not is_blade:
        for d in obj.get('datasets', []):
            datasets.append(d["name"])
    else:
        # Blade-style metadata
        datasets.append("data.csv")
    if is_s3:
        # Download the datasets using the s3 path and dataset names and get the local paths
        paths = fetch_from_s3([os.path.join(os.path.dirname(dataset_metadata), d) for d in datasets])
    else:
        paths = [os.path.join(os.path.dirname(_dataset_metadata), d) for d in datasets]

    return paths, _dataset_metadata


def get_load_dataset_experiment(dataset_paths, dataset_metadata, run_eda=False, dataset_metadata_type="dbench"):
    # Set up the initial experiment to load the dataset
    load_dataset_objective = "Load the dataset and generate summary statistics. "
    load_dataset_steps = f"1. Load the dataset(s) at {[os.path.basename(dp) for dp in dataset_paths]}.\n2. Generate summary statistics for the dataset(s)."
    load_dataset_deliverables = "1. Dataset(s) loaded.\n2. Summary statistics generated."
    if run_eda:
        load_dataset_steps += "\n3. Perform some exploratory data analysis (EDA) on the dataset(s) to get a better understanding of the data."
        load_dataset_deliverables += "\n3. Exploratory data analysis (EDA) performed."
    if dataset_metadata_type == 'blade':
        load_dataset_objective += f"Here is the dataset metadata:\n\n{get_blade_description(dataset_metadata)}"
    else:  # DiscoveryBench-style
        load_dataset_objective += f"Here is the dataset metadata:\n\n{get_dataset_description(dataset_metadata)}"
    load_dataset_experiment = {
        "hypothesis": None,
        "experiment_plan": {
            "objective": load_dataset_objective,
            "steps": load_dataset_steps,
            "deliverables": load_dataset_deliverables
        }
    }
    return load_dataset_experiment

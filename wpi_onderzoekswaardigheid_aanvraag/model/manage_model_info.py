import importlib.resources
import logging
import os
from typing import Dict, Union

import yaml
from azureml.core import Run

from wpi_onderzoekswaardigheid_aanvraag.settings.flags import PipelineFlag

logger = logging.getLogger(__name__)


def load_feature_list():
    folder = "wpi_onderzoekswaardigheid_aanvraag.model.classifier"
    filename = "all_features.yml"
    with importlib.resources.open_binary(folder, filename) as f:
        logger.info(f"Loading feature list from {folder}.{filename}")
        feature_list = yaml.safe_load(f)
        num_cols = feature_list["numerical_cols"]
        cat_cols = feature_list["categorical_cols"]
    return num_cols, cat_cols


def load_selected_features():
    folder = "wpi_onderzoekswaardigheid_aanvraag.model.classifier"
    filename = "selected_features.yml"
    with importlib.resources.open_binary(folder, filename) as f:
        logger.info(f"Loading selected features from {folder}.{filename}")
        features_above_threshold = yaml.safe_load(f)
    return features_above_threshold["features"]


def load_parameter_settings(
    algorithm_type: str, load_grid: Union[bool, PipelineFlag] = False
) -> Dict[str, list]:
    folder = "wpi_onderzoekswaardigheid_aanvraag.model.classifier"

    if load_grid:
        filename = "parameter_grids.yml"
    else:
        filename = "parameters.yml"
    with importlib.resources.open_binary(folder, filename) as f:
        logger.info(f"Loading parameters from {folder}.{filename}...")
        params = yaml.safe_load(f)

    try:
        return params[algorithm_type]
    except KeyError:
        raise ValueError(
            f"No parameter settings present for algorithm type '{algorithm_type}' in file "
            f"'{folder}.{filename}'. Available types are: {list(params.keys())}."
        )


def save_params(params, target_file: str, algorithm_type: str):
    with open(target_file, "r") as f:
        dict_to_write = yaml.safe_load(f)
    params = {k: [v] for k, v in params.items()}
    params = {algorithm_type: params}
    dict_to_write.update(params)
    logger.info("Saving updated optimal parameters...")
    dict_to_yaml(target_file, dict_to_write)


def save_selected_features(features_dict: dict, output_dataset_name: str):
    logger.info("Saving selected features...")
    # Get the experiment run context
    run = Run.get_context()
    mounted_output_dir = run.output_datasets[output_dataset_name]
    os.makedirs(os.path.dirname(mounted_output_dir), exist_ok=True)
    dict_to_yaml(
        f"{mounted_output_dir}/{output_dataset_name}.yml",
        features_dict,
        default_flow_style=False,
    )


def dict_to_yaml(file_path: str, dict_to_write: dict, default_flow_style=None) -> None:
    with open(file_path, "w") as f:
        yaml.safe_dump(dict_to_write, f, default_flow_style=default_flow_style)
        logger.info(f"Wrote dictionary to {file_path}")

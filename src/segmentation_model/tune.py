import csv
import logging
import os
from pathlib import Path
import random

import numpy as np

from src.constants.constants import (
    HYPERPARAM_ARGS,
    DATA_AUGMENT_ARGS,
    ITERATIONS,
    BEST_MODEL_RESULTS,
)
from train import train
from src.utils.utils import get_yaml_files, get_current_timestamp, get_model_size


logger = logging.getLogger(__name__)


def random_search_cv(data_path: Path, model_path: str, iterations: int = ITERATIONS):

    ds_yamls = get_yaml_files(data_path / "train")
    hyperparam_grid = _get_hyperparam_grid()
    best_iou = 0
    std = 0
    model_number = 0

    with_head = "Y" if "without_head" not in data_path.as_posix() else "N"
    timestamp = get_current_timestamp()

    for i in range(iterations):
        random_hyperparams = _select_random_params(hyperparam_grid)
        project = f"{Path(data_path).parts[0]}/runs/segment/kfold_tune/iter{i}"
        random_hyperparams[project] = project

        ious, fold_rows = train(
            ds_yamls=ds_yamls, model_path=model_path, model_kwargs=random_hyperparams
        )
        mean_iou = np.mean(ious)
        _std = np.std(ious, ddof=1)
        if mean_iou > best_iou:
            best_iou = mean_iou
            model_number = i + 1
            std = _std

    mode = "a" if os.path.exists(BEST_MODEL_RESULTS) else "w"
    model_size = get_model_size(model_path)

    with open(BEST_MODEL_RESULTS, mode, newline="") as file:
        writer = csv.writer(file)
        if mode == "w":
            writer.writerow(
                [timestamp, with_head, model_size, model_number, best_iou, std]
            )

    logger.info(
        f"Best model {model_number} with mean IOU {best_iou} and standard deviation {std}"
    )


def _choose_param(param_set):
    if type(param_set) is list:
        if type(param_set[0]) in [float, int]:
            return random.uniform(param_set[0], param_set[1])
        else:
            return random.choice(param_set)

    else:
        return param_set


def _select_random_params(params_dict: dict) -> dict:
    random_params = {}
    for key, value in params_dict.items():
        random_params[key] = (
            _choose_param(value)
            if key != "epochs"
            else random.randint(value[0], value[1])
        )
    return random_params


def _get_hyperparam_grid():
    return {**HYPERPARAM_ARGS, **DATA_AUGMENT_ARGS}

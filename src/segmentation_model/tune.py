"""
Tunes the model using a random search with cross validation.
"""

from datetime import datetime
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import random

import numpy as np
from ultralytics import YOLO

from constants.constants import (
    HYPERPARAM_ARGS,
    DATA_AUGMENT_ARGS,
    ITERATIONS,
)
from segmentation_model.train import train, validate
from utils.utils import get_yaml_files, get_current_timestamp, write_csv_file
from constants.constants import (
    TUNE_TRAIN_RESULTS_HEADER,
    TUNE_TRAIN_RESULTS_FILENAME,
    TUNE_TEST_RESULTS_FILENAME,
    TUNE_TEST_RESULTS_HEADER,
    BEST_MODEL_RESULTS_FILENAME,
    BEST_MODEL_RESULTS_HEADER,
)


logger = logging.getLogger(__name__)


@dataclass
class TuneTrainResults:
    """
    Results from the training iterations.
    """

    best_model_index: int
    mean_ious: list
    std_ious: list
    timestamps: list[str]


@dataclass
class TuneTestingResults:
    """
    Results from testing the best model.
    """

    timestamps: list[str]
    model_path: str
    test_ious: list[float]
    # length should be 5
    test_fold_ious: list[float]
    test_fold_stds: list[float]

    def __init__(self, model_path: str):
        self.timestamps = []
        self.model_path = model_path
        self.test_ious = []
        self.test_fold_ious = []
        self.test_fold_stds = []


def tune(
    data_path: Path,
    model_file_path: str,
    project_path: Path,
    iterations: int = ITERATIONS,
):
    """
    Tune the model for a given number iterations.

    Uses random a hyperparameter search with k-fold cross validation.

    :param data_path: The path to the dataset.
    :param model_file_path: The path to the model.
    :param project_path: The path to store the results.
    :param iterations: The number of iterations to tune for.
    """
    ds_yamls = get_yaml_files(data_path / "train")
    train_results = train_iterations(
        ds_yamls=ds_yamls,
        iterations=iterations,
        project_path=project_path,
        model_file_path=model_file_path,
    )

    iteration_dir = project_path / f"iter{train_results.best_model_index}"
    with_head_flag = "Y" if "without_head" not in data_path.as_posix() else "N"
    with_head_flags = [with_head_flag for i in range(iterations)]

    write_csv_file(
        file_path=project_path / TUNE_TRAIN_RESULTS_FILENAME,
        header=TUNE_TRAIN_RESULTS_HEADER,
        rows=list(
            zip(
                train_results.timestamps,
                [model_file_path for i in range(iterations)],
                with_head_flags,
                train_results.mean_ious,
                train_results.std_ious,
            )
        ),
    )

    test_results = test_iteration(
        iteration_dir=iteration_dir,
        labels_directory=ds_yamls[0].parent.parent.parent / "test/labels",
    )
    write_csv_file(
        file_path=project_path / TUNE_TEST_RESULTS_FILENAME,
        header=TUNE_TEST_RESULTS_HEADER,
        rows=list(
            zip(
                test_results.timestamps,
                with_head_flags,
                [test_results.model_path for i in range(len(test_results.timestamps))],
                test_results.test_fold_ious,
                test_results.test_fold_stds,
            )
        ),
    )

    best_model_iou = train_results.mean_ious[train_results.best_model_index]
    best_model_std = train_results.std_ious[train_results.best_model_index]
    test_mean_ious = np.mean(test_results.test_ious)
    test_std_ious = np.std(test_results.test_ious, ddof=1)

    write_csv_file(
        file_path=project_path / BEST_MODEL_RESULTS_FILENAME,
        header=BEST_MODEL_RESULTS_HEADER,
        rows=[
            [
                train_results.timestamps[train_results.best_model_index],
                with_head_flag,
                model_file_path,
                train_results.best_model_index,
                best_model_iou,
                best_model_std,
                test_mean_ious,
                test_std_ious,
            ]
        ],
    )

    logger.info(
        f"Best model {train_results.best_model_index} with mean IOU {best_model_iou} and standard deviation {best_model_std}"
    )


def train_iterations(
    ds_yamls: list[Path], iterations: int, project_path: Path, model_file_path: str
) -> TuneTrainResults:
    """
    Trains the model on a random set of hyperparameters for a given number of iterations.

    :param ds_yamls: The paths to the dataset yamls to train each iteration on (used for k-fold cross validation).
    :param iterations: The number of iterations to train different models for.
    :param project_path: The path to store the results in.
    :param model_file_path: The path to the model.
    :return: The results training the different iterations.
    """
    hyperparam_grid = _get_hyperparam_grid()
    model_number, best_iou = 0, 0

    fold_ious, fold_stds = [], []
    timestamps = []

    for i in range(iterations):
        timestamps.append(get_current_timestamp())
        random_hyperparams = _select_random_params(hyperparam_grid)
        random_hyperparams["project"] = project_path / f"iter{i}"

        training_results = train(
            ds_yamls=ds_yamls,
            model_path=model_file_path,
            model_kwargs=random_hyperparams,
            run_test=False,
        )
        fold_iou = np.mean(training_results.val_ious)
        fold_std = np.std(training_results.val_ious, ddof=1)

        fold_ious.append(fold_iou)
        fold_stds.append(fold_std)

        if fold_iou > best_iou:
            best_iou = fold_iou
            model_number = i

    return TuneTrainResults(
        best_model_index=model_number,
        mean_ious=fold_ious,
        std_ious=fold_stds,
        timestamps=timestamps,
    )


def test_iteration(iteration_dir: Path, labels_directory: Path) -> TuneTestingResults:
    """
    Run the best iteration's models on the test set.
    :param iteration_dir: The directory for the best iteration.
    :param labels_directory: The directory for the actual labels.
    :return: The results from testing the best iteration.
    """
    test_results = TuneTestingResults(model_path=iteration_dir.as_posix())

    for train_path in os.listdir(iteration_dir):
        test_results.timestamps.append(get_current_timestamp())
        _test_ious = validate(
            model=YOLO(f"{iteration_dir.as_posix()}/{train_path}/weights/best.pt"),
            project=iteration_dir / Path(train_path),
            labels_directory=labels_directory,
            split="test",
        )
        test_results.test_fold_ious.append(np.mean(_test_ious))
        test_results.test_fold_stds.append(np.std(_test_ious, ddof=1))
        test_results.test_ious.extend(_test_ious)

    return test_results


def _get_hyperparam_grid() -> dict:
    """
    Gets hyperparameters and data augmentation ranges.
    """
    return {**HYPERPARAM_ARGS, **DATA_AUGMENT_ARGS}


def _select_random_params(params_dict: dict) -> dict:
    """
    Select random parameters from the param dictionary which contains the ranges to select random parameters from.
    """
    random.seed(datetime.now().timestamp())
    random_params = {}

    for key, value in params_dict.items():
        random_params[key] = (
            _choose_param(value)
            if key != "epochs"
            else random.randint(value[0], value[1])
        )
    return random_params


def _choose_param(
    param_set: list | int | float | str | bool,
) -> int | float | str | bool:
    """
    Choose a param randomly from the given range.
    If only one value is given, then that value is returned.
    """
    if type(param_set) is list:
        if type(param_set[0]) in [float, int]:
            return random.uniform(param_set[0], param_set[1])
        else:
            return random.choice(param_set)

    else:
        return param_set

"""
Methods to get IOUs from previous runs.
"""

from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np

from segmentation_model.iou import calculate_ious


@dataclass
class IouResult:
    run_id: Path
    mean_iou: float
    std_iou: float


def get_results_nokfolds(predictions_path: Path, labels_dir: Path) -> IouResult:
    """
    Gets IOU results from a run that didn't use k-fold cross validation.

    :param predictions_path: The path to the predictions.json file.
    :param labels_dir: The directory path to the actual labels.
    :return: The IOUs of each prediction from the run.
    """

    ious = calculate_ious(
        predictions_path=predictions_path, labels_directory=labels_dir
    )
    mean_iou = np.mean(ious)
    std_iou = np.std(ious, ddof=1)
    iou_result = IouResult(
        run_id=predictions_path.parent, mean_iou=mean_iou, std_iou=std_iou
    )

    return iou_result


def get_results_kfolds(run_dir: Path, data_train: Path) -> list[IouResult]:
    """
    Gets the IOU results from a run that used k-fold cross validation.

    :param run_dir: The directory that contains the run results of all the splits.
    :param data_train: The dataset training directory.
    :return: The mean IOUs of each of the folds.
    """

    ious = []
    rows = []

    for i in range(1, 6):
        if i != 1:
            predictions_path = run_dir / f"train{i}/predictions.json"
        else:
            predictions_path = run_dir / f"train/predictions.json"

        labels_dir = data_train / f"split_{i}/val/labels"
        _ious = calculate_ious(
            predictions_path=predictions_path, labels_directory=labels_dir
        )
        ious.extend(_ious)
        rows.append(
            IouResult(
                run_id=predictions_path.parent,
                mean_iou=np.mean(_ious),
                std_iou=np.std(_ious, ddof=1),
            )
        )

    iou_result = IouResult(
        run_id=run_dir, mean_iou=np.mean(ious), std_iou=np.std(ious, ddof=1)
    )
    rows.append(iou_result)

    return rows


def get_tune_results_kfolds(iterations_dir: Path, data_train: Path) -> list[IouResult]:
    """
    Gets the IOU results for each of the tuning iterations.

    :param iterations_dir: The directory containing the tuning iterations.
    :param data_train: The directory to the dataset training set.
    :return: The IOU results of each iteration.
    """

    rows = []

    for iteration_dir in os.listdir(iterations_dir):
        iteration_ious = []
        for i in range(1, 6):
            if i != 1:
                predictions_path = iteration_dir / f"train{i}/predictions.json"
                labels_dir = data_train / f"split_{i}/train/labels"
                _ious = calculate_ious(
                    predictions_path=predictions_path, labels_dir=labels_dir
                )
                iteration_ious.extend(_ious)

        iou_result = IouResult(
            run_id=iteration_dir,
            mean_iou=np.mean(iteration_ious),
            std_iou=np.std(iteration_ious, ddof=1),
        )
        rows.append(iou_result)

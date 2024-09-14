"""
Methods to get IOUs from previous runs.
"""

from dataclasses import dataclass
import logging
import os
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from segmentation_model.iou import calculate_ious_json
from segmentation_model.validate import validate
from utils.utils import write_csv_file, get_current_timestamp

logger = logging.getLogger(__name__)


@dataclass
class IouResult:
    run_id: Path
    mean_iou: float
    std_iou: float


def get_test_results(project: Path, model_path: str, input_directory: Path, labels_dir: Path, results_path: Path) -> IouResult:
    """
    Gets IOU results from a run that didn't use k-fold cross validation.

    :param predictions_path: The path to the predictions.json file.
    :param labels_dir: The directory path to the actual labels.
    :return: The IOUs of each prediction from the run.
    """
    model = YOLO(model_path)

    timestamp = get_current_timestamp()
    _test_ious = validate(
        model=model,
        project=project,
        name="test",
        input_directory=input_directory,
        labels_directory=labels_dir
    )
    header = ["Timestamp", "Model", "Test Mean IOU", "Test StDev IOU"]
    rows = [[
        timestamp, model_path, np.mean(_test_ious), np.std(_test_ious, ddof=1)
    ]]

    write_csv_file(
        file_path=results_path,
        header=header,
        rows=rows
    )


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
        _ious = calculate_ious_json(labels_directory=labels_dir, predictions_path=predictions_path)
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


def get_tune_results_kfolds(iterations_dir: Path, dataset_path: Path, end_iter: int, start_iter: int = 0) -> list[IouResult]:
    """
    Gets the IOU results for each of the tuning iterations.

    :param iterations_dir: The directory containing the tuning iterations.
    :param dataset_path: The directory to the dataset.
    :return: The IOU results of each iteration.
    """

    rows = []
    best_iou = None
    best_std = None
    best_index = None

    for index in range(start_iter, end_iter + 1):
        mean_ious = []
        std_ious = []
        iteration_dir = iterations_dir / f"iter{index}"
        for i in range(1, 6):

            train_dir = f"train{i}" if i != 1 else "train"
            if os.path.exists(iteration_dir / f"{train_dir}/val"):
                os.system(f"rm -rf {iteration_dir / f'{train_dir}/val'}")

            labels_dir = dataset_path / f"train/split_{i}/val/labels"
            input_dir = dataset_path / f"train/split_{i}/val/images"
            ious = validate(
                model=YOLO(iteration_dir / f"{train_dir}/weights/best.pt"),
                project=iteration_dir / train_dir,
                name="val",
                input_directory=input_dir,
                labels_directory=labels_dir
            )
            mean_ious.append(np.mean(ious))
            std_ious.append(np.std(ious, ddof=1))

        iter_mean_iou = np.mean(mean_ious)
        iter_mean_std = np.mean(std_ious)
        if (best_index is not None and iter_mean_iou > best_iou) or best_index is None:
            best_index = index
            best_iou = iter_mean_iou
            best_std = iter_mean_std

        rows.append([index, iter_mean_iou, iter_mean_std])

    csv_header = ["Model Number", "Mean IOU", "Mean Std"]

    write_csv_file(
        file_path=iterations_dir / "tuning_validation_results.csv",
        header=csv_header,
        rows=rows
    )
    logger.info(f"Best model: {best_index}, Mean IOU: {best_iou}, Mean Std: {best_std}")

    best_iter_path = iterations_dir / f"iter{best_index}"
    test_fold_ious = []
    test_fold_stds = []

    for i in range(1, 6):
        train_dir = f"train{i}" if i != 1 else "train"
        if os.path.exists(best_iter_path / f"{train_dir}/test"):
            os.system(f"rm -rf {best_iter_path / f'{train_dir}/test'}")

        labels_dir = dataset_path / f"test/labels"
        input_dir = dataset_path / f"test/images"
        ious = validate(
            model=YOLO(best_iter_path / f"{train_dir}/weights/best.pt"),
            project=best_iter_path / train_dir,
            name="test",
            input_directory=input_dir,
            labels_directory=labels_dir
        )
        test_fold_ious.append(np.mean(ious))
        test_fold_stds.append(np.std(ious, ddof=1))

    test_mean_iou = np.mean(test_fold_ious)
    test_mean_std = np.mean(test_fold_stds)

    logger.info(f"Test results for model: {best_index}, Mean IOU: {test_mean_iou}, Mean Std: {test_mean_std}")

    write_csv_file(
        file_path=iterations_dir / "best_model.csv",
        header=["Model Number", "Val Mean IOU", "Val Mean Std", "Test Mean IOU", "Test Mean Std"],
        rows=[[best_index, best_iou, best_std, test_mean_iou, test_mean_std]]
    )

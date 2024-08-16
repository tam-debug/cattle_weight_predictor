"""
Train a segmentation model.
"""

import logging
import numpy as np
from pathlib import Path

from ultralytics import YOLO

from src.constants.constants import TRAINING_RESULTS_NAME, TRAINING_RESULTS_HEADER
from src.utils.utils import get_current_timestamp, get_model_size, write_csv_file
from validate import validate

logger = logging.getLogger(__name__)


def get_train_project(data_path: Path) -> str:
    """
    Gets a project directory for training.

    :param data_path: The path of the dataset YAML.
    :return: The project path that can be used during model training.
    """
    return f"{Path(data_path).parts[0]}/runs/segment/kfold"


def write_training_results(
    timestamps: list[float], ious: list[list[float]], model_size: str
):
    """
    Writes results from a model training to a CSV file.

    :param timestamps: The timestamps of the training results.
    :param ious: The IOUs of the training results.
    :param model_size: The size of the model that was trained.
    """
    rows = []
    for i in range(len(ious)):
        ious_fold = ious[i]
        timestamp = timestamps[i]

        fold_iou = np.mean(ious_fold)
        fold_std = np.std(ious_fold, ddof=1)

        rows.append([timestamp, model_size, fold_iou, fold_std])

    results_path = Path(f"results/{TRAINING_RESULTS_NAME}")

    write_training_results(
        rows=rows, headers=TRAINING_RESULTS_HEADER, results_path=results_path
    )


def train(
    ds_yamls: list[Path], model_path: str, model_kwargs: dict
) -> tuple[list[float], list[list]]:
    """
    Trains the model on the folds specified in the YAML files.

    :param ds_yamls: The YAMLs of the datasets to train on.
    :param model_path: The path of the model e.g. "yolov8m-seg.pt"
    :param model_kwargs: The key word arguments to pass to the model.
    :return: The IOUs for each prediction, Training results rows of format (timestamp, fold average IOU, fold std IOU)
    """

    ious = []
    timestamps, fold_ious, fold_stds = [], [], []

    for i in len(ds_yamls):
        timestamp = get_current_timestamp()
        dataset_yaml = ds_yamls[i]
        model = YOLO(model_path)
        model.train(data=dataset_yaml, **model_kwargs)

        val_labels_dir = ds_yamls[0].parent / "labels"
        _ious = validate(
            data_path=dataset_yaml, model=model, labels_directory=val_labels_dir
        )
        ious.append(_ious)

        timestamps.append(timestamp)
        fold_ious.append(np.mean(_ious))
        fold_stds.append(np.std(_ious, ddof=1))

    logger.debug(
        f"""Results
    {ious=}
    """
    )
    rows = list(zip(timestamps, fold_ious, fold_stds))

    return ious, rows

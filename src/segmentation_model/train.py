"""
Train a segmentation model.
"""

from dataclasses import dataclass
import logging
import numpy as np
from pathlib import Path

from ultralytics import YOLO

from constants.constants import (
    TRAINING_RESULTS_HEADER,
)
from utils.utils import get_current_timestamp, write_csv_file
from segmentation_model.validate import validate

logger = logging.getLogger(__name__)


@dataclass
class TrainingResults:
    timestamps: list[str]
    model_path: str
    val_ious: list[float]
    test_ious: list[float]
    # length should be 5
    val_fold_ious: list[float]
    val_fold_stds: list[float]
    test_fold_ious: list[float]
    test_fold_stds: list[float]

    def __init__(self, model_path: str):
        self.timestamps = []
        self.model_path = model_path
        self.val_ious = []
        self.test_ious = []
        self.val_fold_ious = []
        self.val_fold_stds = []
        self.test_fold_stds = []
        self.test_fold_ious = []
        self.test_fold_stds = []


def train(
    ds_yamls: list[Path], model_path: str, model_kwargs: dict, run_test: bool = True
) -> TrainingResults:
    """
    Trains the model on the folds specified in the YAML files.

    :param ds_yamls: The YAMLs of the datasets to train on.
    :param model_path: The path of the model e.g. "yolov8m-seg.pt"
    :param model_kwargs: The key word arguments to pass to the model.
    :param run_test: Whether to run the data on the test set.
    :return: The IOUs for each prediction, Training results rows of format (timestamp, fold average IOU, fold std IOU)
    """

    training_results = TrainingResults(model_path=model_path)

    for i in range(len(ds_yamls)):
        dataset_yaml = ds_yamls[i]

        model_kwargs["single_cls"] = True

        training_results.timestamps.append(get_current_timestamp())

        model = YOLO(model_path)
        train_res = model.train(data=dataset_yaml, **model_kwargs)

        _val_ious = validate(
            model=model,
            project=train_res.save_dir,
            name="val",
            input_directory=ds_yamls[i].parent / "val/images",
            labels_directory=ds_yamls[i].parent / "val/labels",
        )
        if len(_val_ious) > 0:
            training_results.val_ious.extend(_val_ious)
            training_results.val_fold_ious.append(np.mean(_val_ious))
            training_results.val_fold_stds.append(np.std(_val_ious, ddof=1))
        else:
            training_results.val_fold_ious.append(0)
            training_results.val_fold_stds.append(0)

        if run_test:
            _test_ious = validate(
                model=model,
                project=train_res.save_dir,
                name="test",
                input_directory=ds_yamls[i].parent.parent.parent / "test/images",
                labels_directory=ds_yamls[i].parent.parent.parent / "test/labels",
            )

            training_results.test_ious.extend(_test_ious)

            training_results.test_fold_ious.append(np.mean(_test_ious))
            training_results.test_fold_stds.append(np.std(_test_ious, ddof=1))

    logger.debug(
        f"""Results
{training_results.val_ious=}
"""
    )

    return training_results


def write_training_results(file_path: Path, training_results: TrainingResults):
    """
    Write the training results to a file.

    :param file_path: The file to write to.
    :param training_results: The training results object.
    """
    model_path_column = [
        training_results.model_path for i in range(len(training_results.timestamps))
    ]
    rows = list(
        zip(
            training_results.timestamps,
            model_path_column,
            training_results.val_fold_ious,
            training_results.val_fold_stds,
            training_results.test_fold_ious,
            training_results.test_fold_stds,
        )
    )

    write_csv_file(file_path=file_path, header=TRAINING_RESULTS_HEADER, rows=rows)

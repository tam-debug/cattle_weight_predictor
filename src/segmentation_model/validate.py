"""
Function for validating the segmentation model against a given dataset.
"""

from pathlib import Path

from ultralytics import YOLO

from iou import calculate_ious
from src.constants.constants import PREDICTION_FILENAME


def validate(data_path: Path, model: YOLO, labels_directory: Path) -> list[float]:
    """
    Runs the segmentation model against the dataset YAML and calculates the IOU for the masks.

    :param data_path: The dataset YAML file path that contains the validation directory "val"
    :param model: The model to run the dataset against.
    :param labels_directory: The directory that contains the label text files.
    :return: The IOUs from the validation.
    """
    results = model.val(data=data_path, save_json=True)

    predictions_path = results.save_dir / PREDICTION_FILENAME
    ious = calculate_ious(
        predictions_path=predictions_path, labels_directory=labels_directory
    )

    return ious

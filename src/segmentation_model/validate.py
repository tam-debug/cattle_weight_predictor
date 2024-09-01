import os
from pathlib import Path

from constants.constants import PREDICTION_FILENAME
from segmentation_model.iou import calculate_ious


def validate(
    model, project: Path, labels_directory: Path, split: str = "val"
) -> list[float]:
    """
    Validate the model against the chosen split.

    :param model: The model to run validation on.
    :param project: The directory the validation results will be saved.
    :param labels_directory: The directory where the labels are stored.
    :param split: The split to run validation on e.g. "val", "test"
    :return: The IOUs from the validation.
    """
    test_results = model.val(split=split, save_json=True, name=split, project=project)

    test_predictions_path = test_results.save_dir / PREDICTION_FILENAME
    if os.path.exists(test_predictions_path):
        ious = calculate_ious(
            predictions_path=test_predictions_path, labels_directory=labels_directory
        )
    else:
        ious = []

    return ious

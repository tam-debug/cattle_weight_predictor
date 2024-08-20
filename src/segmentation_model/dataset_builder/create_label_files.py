"""
This creates with head and without label (.txt) files from the json annotation files.
"""

import os
from pathlib import Path

import numpy as np

from src.constants.constants import (
    IMAGES_DIR,
    LABELS_DIR_WITH_HEAD,
    LABELS_DIR_WITHOUT_HEAD,
)
from src.utils.utils import parse_json


def create_label_files(
    json_data_dir: str,
    start: int = None,
    end: int = None,
    with_head_labels_dir: str = LABELS_DIR_WITH_HEAD,
    without_head_labels_dir: str = LABELS_DIR_WITHOUT_HEAD,
):
    """
    Creates the label files.

    :param json_data_dir: The directory where the JSON annotation files are stored.
    :param with_head_labels_dir: The directory where the labels that include the cattle's heads will be stored.
    :param without_head_labels_dir: The directory where the labels that exclude the cattle's heads will be stored.
    """
    _create_directories([with_head_labels_dir, without_head_labels_dir])
    if start is None:
        json_files = [
            Path(pos_json)
            for pos_json in os.listdir(json_data_dir)
            if pos_json.endswith(".json")
        ]
    else:
        json_files = [Path(f"{number:05d}.json") for number in range(start, end + 1)]

    for json_file in json_files:
        label_file = f"{json_file.stem}.txt"
        create_label_file(
            json_path=Path(json_data_dir) / json_file,
            with_head_labels_path=Path(with_head_labels_dir) / label_file,
            without_head_labels_path=Path(without_head_labels_dir) / label_file,
        )


def create_label_file(
    json_path: Path, with_head_labels_path: Path, without_head_labels_path: Path
) -> None:
    """
    Creates the two labels files (one for with head and one for without) from a json file.
    """
    with_head_points = []
    without_head_points = []
    if not os.path.exists(json_path):
        image_width, image_height = 1, 1
    else:
        parsed_json = parse_json(json_path)

        image_width = parsed_json["imageWidth"]
        image_height = parsed_json["imageHeight"]

        for shape in parsed_json["shapes"]:
            group_id = shape["group_id"]
            points = shape["points"]

            if group_id == 0:
                without_head_points.append(points)
            elif group_id == 1:
                with_head_points.append(points)
            elif group_id == 2:
                without_head_points.append(points)
                with_head_points.append(points)
            else:
                raise ValueError(
                    f"Group id should be 0, 1 or 2 and not {group_id} for json path {json_path}"
                )

    _write_label_file(
        with_head_labels_path, with_head_points, image_width, image_height
    )
    _write_label_file(
        without_head_labels_path, without_head_points, image_width, image_height
    )


def _create_directories(directories_to_create: list) -> None:
    """
    Creates the directories that will store the images and their labels.
    """
    for directory in directories_to_create:
        if not os.path.exists(directory):
            os.makedirs(directory)


def _write_label_file(
    label_path: Path, shapes: list[list], image_width: int, image_height: int
) -> None:
    """
    Writes the different sets of points to the label file in a normalised form.
    """
    with open(label_path, "w") as label_file:
        for points in shapes:
            points = np.array(points, float)
            points[:, 0] /= image_width
            points[:, 1] /= image_height

            label_file.write("0")
            for x, y in points:
                label_file.write(f" {x} {y}")
            label_file.write("\n")

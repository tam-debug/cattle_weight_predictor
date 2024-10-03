"""
Utility methods.
"""

import csv
import cv2
from datetime import datetime
import json
import os
from pathlib import Path

import numpy as np
import torch

from constants.constants import YOLO_MODEL_SIZES


def get_yolo_seg_model_name(size: str) -> str:
    """
    Gets the name of the YOLOv8 segmentation model for the given size.

    :param size: The size of the YOLO model e.g. s, m, l
    :return: The name of the YOLOv8 segmentation model for the particular size.
    """
    if size not in YOLO_MODEL_SIZES:
        raise ValueError(
            f"Model size must be either {' '.join(YOLO_MODEL_SIZES)} and not {size}"
        )
    else:
        return f"yolov8{size}-seg.pt"


def get_model_size(model_name) -> str:
    index = model_name.find("yolov8")
    if index != -1:
        return model_name[index + len("yolov8")]
    else:
        return model_name


def get_yaml_files(directory: Path) -> list[Path]:
    """
    Recursively sSearches and returns a list of yaml files.

    The list will be empty if there are no yaml files.

    :param directory: The directory to search for the yaml files in.
    :return: List of the yaml file paths.
    """
    return list(directory.rglob("*.yaml"))


def get_current_timestamp() -> str:
    """
    Gets the current timestamp as a string.

    :return: The current timestamp in the string format YYYY-MM-DD HH:MM:SS
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_image_id(number: int) -> str:
    """
    Left pads the image number with 0s so its digit length is 5.

    :param number: The number to format.
    :return: The image id as a 5-digit number.
    """
    formatted_number = f"{number:05}"
    return formatted_number


def write_csv_file(file_path: Path, rows: list[list], header=None):
    """
    Writes rows to a csv file.

    If the file doesn't exist a header is added (if supplied).
    If the file exists, the rows are appended.

    :param file_path: The path of the csv file.
    :param rows: The rows to write to the csv file.
    :param header: The header of the csv.
    """
    mode = "a" if os.path.exists(file_path) else "w"
    with open(file_path, mode, newline="") as file:
        writer = csv.writer(file)
        if header is not None and mode == "w":
            writer.writerow(header)
            writer.writerows(rows)
        else:
            writer.writerows(rows)


def parse_json(file_path: Path) -> dict:
    """
    Parses the given json file into a dictionary format.
    :param file_path: The json file path.
    :return: The parsed JSON data as a dictionary.
    """
    if file_path.suffix != ".json":
        raise ValueError(f"File path must end with .json and not {file_path}")

    with open(file_path) as file:
        parsed_json = json.load(file)

    return parsed_json


def show_mask(mask: np.ndarray):
    """
    Shows the mask in a new window.
    :param mask: The mask to display.
    """
    # output_mask_normalized = np.uint8(output_mask_normalized)
    cv2.imshow("Output Mask", mask.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)


def resize_mask(masks: torch.tensor, height: int, width: int) -> np.ndarray:
    """
    Resizes the segmentation mask to the specified size.
    """
    resized_masks = torch.nn.functional.interpolate(
        masks.unsqueeze(1), size=(height, width), mode="nearest"
    ).squeeze(1)
    resized_masks_np = resized_masks.cpu().numpy()
    return resized_masks_np

def colour_depth_maps(directory: Path, save_dir: Path):

    for i in range(1, 104):

        depth_map = cv2.imread(directory / f"{i}.png", cv2.IMREAD_GRAYSCALE)
        # Normalize the depth map to the range [0, 255]
        depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)

        # Convert to 8-bit
        depth_map_normalized = depth_map_normalized.astype(np.uint8)

        # Apply a color map
        colored_depth_map = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)

        # Display the result
        cv2.imwrite(save_dir / f"{i}.png", colored_depth_map)

"""
Loads or generates the segmentation masks from the segmentation model.
"""

import os.path
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from typing import Union
from ultralytics import YOLO
from ultralytics.engine.results import Results
from utils.utils import parse_json


@dataclass
class ImageSegmentationMasks:
    masks: torch.Tensor
    original_hw: tuple[int, int]


def load_seg_mask_tensors(
    tensor_dir: Path, start_num: int, end_num: int
) -> dict[int, ImageSegmentationMasks]:
    """
    Load the segmentation masks that were stored as PyTorch tensors in .pt files.
    :param tensor_dir: The directory that contains the tensor files.
    :param start_num: The start number of the file to load.
    :param end_num: The end number of the file to load.
    :return: The segmentation masks with their corresponding image number.
    """
    image_masks = {}

    for i in range(start_num, end_num + 1):
        raw = torch.load(tensor_dir / f"{i}.pt")

        image_masks[i] = ImageSegmentationMasks(
            masks=raw["masks"], original_hw=raw["original_hw"]
        )

    return image_masks


def load_prelabel_masks(
    directory: Path, start_num: int, end_num: int
) -> dict[int, ImageSegmentationMasks]:
    """
    Load the AnyLabeling JSON segmentation files as a binary mark.

    :param directory: The directory that contains the files to load.
    :param start_num: The start number of the file to load.
    :param end_num: The end number of the file to load.
    :return: The segmentation masks.
    """

    seg_masks = {}
    for i in range(start_num, end_num + 1):
        dictionary = parse_json(directory / f"{i}.json")
        points = dictionary["shapes"][0]["points"]
        polygon_points = np.array(points)
        polygon_points = np.round(polygon_points).astype(np.int32)
        height = dictionary["imageHeight"]
        width = dictionary["imageWidth"]
        mask = np.zeros((height, width), dtype=np.uint8)

        # Reshape the polygon points to match the format expected by fillPoly (1, n, 2)
        polygon_points = polygon_points.reshape((-1, 1, 2))

        # Fill the polygon in the mask with 1s (set color as 1 for binary mask)
        cv2.fillPoly(mask, [polygon_points], 1)
        mask = torch.tensor(mask)
        mask = mask.unsqueeze(0)

        seg_masks[i] = ImageSegmentationMasks(masks=mask, original_hw=(height, width))

    return seg_masks


def generate_masks(
    model_path: str,
    images: Union[Path, list[Path]],
    project: Path = None,
    name: str = None,
    save_tensors: bool = False,
) -> dict[int, ImageSegmentationMasks]:
    """
    Generates the segmentation masks for the given images.

    The segmentation masks are Pytorch tensors.

    :param model_path: The model to use to generate the segmentation masks.
    :param images: The image directory or list of images to generate the segmentation masks for.
    :param project: The parent directory where the prediction results are stored.
    :param name: The directory under the project where the prediction results are stored.
    :param save_tensors: Whether to save the segmentation mask tensors as .pt files.
    :return: The image path, and its corresponding segmentation masks.
    """
    image_masks = {}
    prediction_results = _predict(model_path, images, project=project, name=name)
    for result in prediction_results:
        image_path = Path(result.path)
        image_number = int(image_path.stem)
        masks = result.masks.data
        original_hw = result.orig_shape

        seg_masks = ImageSegmentationMasks(masks, original_hw)
        image_masks[image_number] = seg_masks

    if save_tensors:
        _save_seg_masks(project=project / name, masks=image_masks)

    return image_masks


def _predict(
    model_path: str,
    source: Union[Path, list[Path]],
    project: Path = None,
    name: str = None,
) -> list[Results]:
    """
    Generates segmentation masks for the given image using the given model.
    """
    model = YOLO(model_path)
    if project:
        results = model.predict(source=source, project=project, name=name, save=True)
    else:
        results = model.predict(source)
    return results


def _save_seg_masks(project: Path, masks: dict[int, ImageSegmentationMasks]):
    """
    Save the segmentation masks as PyTorch tensors in .pt files.
    """

    save_dir = project / "tensors"

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for image_number, image_seg_masks in masks.items():
        save_path = save_dir / f"{image_number}.pt"
        torch.save(
            {
                "masks": image_seg_masks.masks,
                "original_hw": image_seg_masks.original_hw,
            },
            save_path,
        )

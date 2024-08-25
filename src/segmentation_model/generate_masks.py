"""
Generates the segmentation masks from the segmentation model.
"""

from dataclasses import dataclass
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results

@dataclass
class ImageSegmentationMasks:
    masks: torch.Tensor
    original_hw: tuple[int, int]

def generate_masks(
    model_path: str, images: Path | list[Path]
) -> dict[Path, ImageSegmentationMasks]:
    """
    Generates the segmentation masks for the given images.

    The segmentation masks are Pytorch tensors.

    :param model_path: The model to use to generate the segmentation masks.
    :param images: The image directory or list of images to generate the segmentation masks for.
    :return: The image path, and its corresponding segmentation masks.
    """
    image_masks = {}
    prediction_results = predict(model_path, images)
    for result in prediction_results:
        image_path = Path(result.path)
        masks = result.masks.data
        original_hw = result.orig_shape

        seg_masks = ImageSegmentationMasks(masks, original_hw)
        image_masks[image_path] = seg_masks

        # if image_masks.get(image_path) is None:
        # else:
        #     for i in range(masks):
        #         image_masks[image_path].append(seg_masks)

    return image_masks


def predict(model_path: str, source: Path | list[Path]) -> list[Results]:
    """
    Generates segmentation masks for the given image using the given model.

    :param model_path: The path of the model to use.
    :param source: The path of the image (or list of paths) or image directory to generate the segmentation mask on.
    :return: List of the results, of which the segmentation masks are stored.
    """
    model = YOLO(model_path)
    results = model.predict(source)
    return results



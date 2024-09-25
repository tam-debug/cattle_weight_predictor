"""
Generates the segmentation masks from the segmentation model.
"""
import os.path
from dataclasses import dataclass
from pathlib import Path

import torch
from typing import Union
from ultralytics import YOLO
from ultralytics.engine.results import Results

@dataclass
class ImageSegmentationMasks:
    masks: torch.Tensor
    original_hw: tuple[int, int]

def generate_masks(
    model_path: str, images: Union[Path, list[Path]], project: Path = None, name: str = None, save_tensors: bool = False
) -> dict[Path, ImageSegmentationMasks]:
    """
    Generates the segmentation masks for the given images.

    The segmentation masks are Pytorch tensors.

    :param model_path: The model to use to generate the segmentation masks.
    :param images: The image directory or list of images to generate the segmentation masks for.
    :return: The image path, and its corresponding segmentation masks.
    """
    image_masks = {}
    prediction_results = predict(model_path, images, project=project, name=name)
    for result in prediction_results:
        image_path = Path(result.path)
        masks = result.masks.data
        original_hw = result.orig_shape

        seg_masks = ImageSegmentationMasks(masks, original_hw)
        image_masks[image_path] = seg_masks

    if save_tensors:
        _save_seg_masks(
            project=project / name,
            masks=image_masks
        )

    return image_masks

def load_seg_mask_tensors(tensor_dir: Path) -> dict[Path, ImageSegmentationMasks]:
    image_masks = {}

    for tensor_path in os.listdir(tensor_dir):
        raw = torch.load(tensor_dir / tensor_path)

        image_masks[Path(tensor_path).stem] = ImageSegmentationMasks(
            masks=raw["masks"],
            original_hw=raw["original_hw"]
        )

    return image_masks

def _save_seg_masks(project: Path, masks: dict[Path, ImageSegmentationMasks]):

    save_dir = project / "tensors"

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for image_path, image_seg_masks in masks.items():
        save_path = save_dir / f"{image_path.stem}.pt"
        torch.save({
            "masks": image_seg_masks.masks,
            "original_hw": image_seg_masks.original_hw
        }, save_path)


def predict(model_path: str, source: Union[Path, list[Path]], project: Path = None, name: str = None) -> list[Results]:
    """
    Generates segmentation masks for the given image using the given model.

    :param model_path: The path of the model to use.
    :param source: The path of the image (or list of paths) or image directory to generate the segmentation mask on.
    :return: List of the results, of which the segmentation masks are stored.
    """
    model = YOLO(model_path)
    if project:
        results = model.predict(source=source, project=project, name=name, save=True)
    else:
        results = model.predict(source)
    return results



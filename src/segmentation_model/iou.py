"""
Calculates the IOU for the prediction masks against the ground truth masks.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pycocotools import mask as mask_utils

from constants.constants import CONFIDENCE_SCORE_THRESHOLD
from utils.utils import format_image_id, parse_json

@dataclass
class PredictionResult:
    image_id: str
    masks: list
    mask_size: tuple[int, int]
def calculate_ious(labels_directory: Path, prediction_results: list[PredictionResult]) -> list[float]:
    """
    Calculates the IOU for each prediction mask against its ground truth mask.

    :param labels_directory: The directory containing the label text files.
    :return: The IOUs for the prediction masks.
    """
    ious = []

    for prediction_result in prediction_results:
        if prediction_result.image_id == "00342":
            a = 1
        pred_masks = prediction_result.masks
        mask_size = pred_masks[0].shape if len(pred_masks) > 0 else prediction_result.mask_size
        gt_mask_path = labels_directory / f"{prediction_result.image_id}.txt"
        gt_masks = _ground_truth_to_mask(gt_mask_path, mask_size)

        _ious = _calculate_ious(pred_masks, gt_masks)
        ious.extend(_ious)
    return ious

def calculate_ious_json(labels_directory: Path, prediction_masks: dict = None, predictions_path: Path = None) -> list[float]:
    """
    Calculates the IOU for each prediction mask against its ground truth mask.

    :param predictions_path: The path to the predictions JSON file.
    :param labels_directory: The directory containing the label text files.
    :return: The IOUs for the prediction masks.
    """
    ious = []
    prediction_masks = prediction_masks if prediction_masks else _get_masks_from_json_predictions(predictions_path)

    for image_id, pred_masks in prediction_masks.items():
        gt_mask_path = labels_directory / f"{format_image_id(image_id)}.txt"
        gt_masks = _ground_truth_to_mask(gt_mask_path, pred_masks[0].shape)

        _ious = _calculate_ious(pred_masks, gt_masks)
        ious.extend(_ious)
    return ious


def _get_masks_from_json_predictions(json_file_path: Path) -> dict:
    """
    Loads the prediction masks from the JSON file.
    """
    predictions = parse_json(json_file_path)

    image_masks = {}

    for image_pred in predictions:
        image_id = image_pred["image_id"]
        mask = image_pred.get("segmentation")
        mask = mask_utils.decode(mask)
        confidence_score = image_pred["score"]

        if confidence_score > CONFIDENCE_SCORE_THRESHOLD:
            if image_masks.get(image_id) is None:
                image_masks[image_id] = [mask]
            else:
                image_masks[image_id].append(mask)

    return image_masks


def _ground_truth_to_mask(
    file_path: Path, img_shape: tuple[int, int]
) -> list[np.ndarray]:
    """
    Load the label text file that contains a ground truth mask.
    """

    masks = []
    img_height, img_width = img_shape[:2]

    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.strip():
                mask = np.zeros(img_shape[:2], dtype=np.uint8)
                coords = np.array([float(c) for c in line.split()[1:]]).reshape(-1, 2)
                coords[:, 0] *= img_width  # Scale x coordinates
                coords[:, 1] *= img_height  # Scale y coordinates
                coords = coords.astype(int)

                # Fill the polygon formed by the boundary coordinates
                cv2.fillPoly(mask, [coords], 1)
                masks.append(mask)

    return masks


def show_mask(mask: np.ndarray):

    # resized_image = cv2.resize(mask (width, height), interpolation=cv2.INTER_NEAREST)

    # Display the original and resized images
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(mask, cmap="gray")
    plt.axis("off")

    # plt.subplot(1, 2, 2)
    # plt.title('Resized Image')
    # plt.imshow(resized_image, cmap='gray')
    # plt.axis('off')

    plt.show()


def _separate_instances(mask):
    """
    Separates instances in one image mask into separate masks.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    masks = []
    for i in range(1, num_labels):  # Start from 1 to skip the background
        instance_mask = (labels == i).astype(np.uint8) * 255
        masks.append(instance_mask)

    for m in masks:
        resized_image = cv2.resize(mask, (1280, 720), interpolation=cv2.INTER_NEAREST)
        show_mask(resized_image)
    return masks


def _calculate_ious(pred_masks, gt_masks) -> list[float]:
    """
    Calculates the IOUs for the prediction masks and the ground truth masks.
    """
    ious = []
    for pred_mask in pred_masks:
        best_iou = 0
        for gt_mask in gt_masks:
            iou = _calculate_iou(pred_mask, gt_mask)
            if iou > best_iou:
                best_iou = iou
        ious.append(best_iou)
    return ious


def _calculate_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Calculates the IOU for a prediction mask, given its ground truth mask.
    """

    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    iou = intersection / union if union != 0 else 0
    return iou

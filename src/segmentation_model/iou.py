"""
Calculates the IOU for the prediction masks against the ground truth masks.
"""

from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

from src.constants.constants import CONFIDENCE_SCORE_THRESHOLD
from src.utils.utils import format_image_id, parse_json


def calculate_ious(predictions_path: Path, labels_directory: Path) -> list[float]:
    """
    Calculates the IOU for each prediction mask against its ground truth mask.

    :param predictions_path: The path to the predictions JSON file.
    :param labels_directory: The directory containing the label text files.
    :return: The IOUs for the prediction masks.
    """
    ious = []
    prediction_masks = _get_masks_from_json_predictions(predictions_path)

    for image_id, pred_masks in prediction_masks.items():
        gt_mask_path = labels_directory / f"{format_image_id(image_id)}.txt"
        combined_gt_masks = _ground_truth_to_mask(gt_mask_path, pred_masks[0].shape)
        gt_masks = _separate_instances(combined_gt_masks)

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


def _ground_truth_to_mask(file_path, img_shape):
    """
    Load the label text file that contains a ground truth mask.
    """

    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    img_height, img_width = img_shape[:2]

    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            coords = np.array([float(c) for c in line.split()[1:]]).reshape(-1, 2)
            coords[:, 0] *= img_width  # Scale x coordinates
            coords[:, 1] *= img_height  # Scale y coordinates
            coords = coords.astype(int)

            # Fill the polygon formed by the boundary coordinates
            cv2.fillPoly(mask, [coords], 1)

    return mask


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
    return masks


def _calculate_ious(pred_masks, gt_masks):
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

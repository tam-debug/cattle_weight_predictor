"""
Methods for storing and loading the segmented depth masks with the weights list.
"""

import csv
import cv2
from dataclasses import dataclass
import json
from json import JSONEncoder
import logging
from pathlib import Path
from typing import Union

import numpy as np
import torch

from constants.constants import SCALE_DIMENSION, DEPTH_MASK_DIMENSION
from segmentation_model.generate_masks import generate_masks, load_seg_mask_tensors, load_prelabel_masks
from segmentation_model.dataset_builder.resize_rgb import transform_id_coordinates, RgbTransformParams
from utils.utils import parse_json, resize_mask, show_mask

logger = logging.getLogger(__name__)


@dataclass
class DepthMaskWeight:
    """
    Contains the depth mask and the corresponding weight.
    """

    depth_mask: np.ndarray
    weight: int


@dataclass
class IdMapping:
    """
    Contains the cattle id and corresponding coordinate contained by the cattle.
    """

    x: int
    y: int
    id: str


@dataclass
class SegMaskWeight:
    """
    Contains the binary segmentation mask and the corresponding weight.
    """

    seg_mask: np.ndarray
    weight: int


class NumpyArrayEncoder(JSONEncoder):
    """
    Encoder for writing numpy arrays to JSON format.
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def load_dataset(file_paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads the depth masks and weight from the JSON file.

    :param file_paths: The file path to the JSON file.
    :return: The depth masks and weights in separate numpy arrays.
    """

    depth_masks, weights = [], []

    for file_path in file_paths:
        data = np.load(file_path, allow_pickle=True)
        _depth_masks = data['depth_masks']
        _weights = data['weights']

        if len(weights) == 0:
            weights = _weights
            depth_masks.append(_depth_masks)
        elif not np.array_equal(_weights, weights):
            raise ValueError("Weights are not the same for all datasets.")
        else:
            depth_masks.append(_depth_masks)

    if len(depth_masks) > 1:
        depth_masks = np.stack(depth_masks, axis=1)
    else:
        depth_masks = np.array(depth_masks)
        depth_masks = depth_masks.transpose(1, 0, 2, 3)

    return depth_masks, np.array(weights)


def generate_depth_masks(
    seg_model: str,
    rgb_image_dir: Path,
    depth_image_dir: Path,
    weights_file: Path,
    start_num: int,
    end_num: int,
    rgb_transform_params: RgbTransformParams,
    scale_to_255: bool,
    save_dir: Path = None,
    id_mapping_dir: Path = None,
    seg_mask_dir: Path = None,
    prelabel_mask_dir: Path = None
) -> list[DepthMaskWeight]:
    """
    Generates the segmented depth masks and with the weights list.

    Can optionally save the depth masks and weights to a JSON file.

    :param seg_model: The segmentation model to generate the masks from.
    :param rgb_image_dir: The directory that stores the RGB images.
    :param depth_image_dir: The directory that stores the depth images.
    :param weights_file: The file path that contains the weights for each cattle ID.
    :param save_numpy: Whether to save the generated depth masks in a JSON file.
    :return: The depth masks and weights lists.
    """
    if prelabel_mask_dir:
        seg_masks = load_prelabel_masks(prelabel_mask_dir, start_num=start_num, end_num=end_num)
    elif seg_mask_dir:
        seg_masks = load_seg_mask_tensors(seg_mask_dir, start_num=start_num, end_num=end_num)
    else:
        seg_masks = generate_masks(seg_model, rgb_image_dir)

    weights = _load_weights(weights_file)

    depth_mappings = []

    if id_mapping_dir:
        for image_number, _seg_masks in seg_masks.items():
            seg_masks = _seg_masks.masks
            depth_frame = _load_depth_frame(
                file_path=_get_depth_image_name(
                    image_path=image_number, depth_image_dir=depth_image_dir
                )
            )
            resized_seg_masks = resize_mask(
                seg_masks, height=depth_frame.shape[0], width=depth_frame.shape[1]
            )

            id_mapping_file = id_mapping_dir / f"{image_number}.json"

            seg_mappings = _match_mask_to_weight(
                seg_masks=resized_seg_masks,
                weights=weights,
                id_mapping_file=id_mapping_file,
                original_scale=_seg_masks.original_hw,
                rgb_transform_params=rgb_transform_params
            )

            _depth_mappings = _create_depth_masks(
                seg_mappings=seg_mappings, depth_frame=depth_frame, scale_to_255=scale_to_255
            )
            depth_mappings.extend(_depth_mappings)

    else:
        for image_number, _seg_masks in seg_masks.items():
            weight = weights[str(image_number)]
            depth_frame = _load_depth_frame(
                file_path=_get_depth_image_name(
                    image_path=image_number, depth_image_dir=depth_image_dir
                )
            )
            binary_mask = _seg_masks.masks[0].cpu().numpy()
            depth_mask = _create_depth_mask(
                binary_mask=binary_mask, depth_frame=depth_frame
            )
            show_mask(depth_mask)
            depth_mappings.append(DepthMaskWeight(depth_mask=depth_mask, weight=weight))

    if save_dir:
        _save_depth_masks(depth_mappings=depth_mappings, file_path=save_dir)

    return depth_mappings

def get_depth_masks_mean_and_stddev(
    seg_model: str,
    rgb_image_dir: Path,
    depth_image_dir: Path,
    weights_file: Path,
    start_num: int,
    end_num: int,
    rgb_transform_params: RgbTransformParams,
    id_mapping_dir: Path = None,
    seg_mask_dir: Path = None,
    prelabel_mask_dir: Path = None
) -> list[DepthMaskWeight]:
    if prelabel_mask_dir:
        seg_masks = load_prelabel_masks(prelabel_mask_dir, start_num=start_num, end_num=end_num)
    elif seg_mask_dir:
        seg_masks = load_seg_mask_tensors(seg_mask_dir, start_num=start_num, end_num=end_num)
    else:
        seg_masks = generate_masks(seg_model, rgb_image_dir)

    weights = _load_weights(weights_file)

    depth_values = []

    if id_mapping_dir:
        for image_number, _seg_masks in seg_masks.items():
            seg_masks = _seg_masks.masks
            depth_frame = _load_depth_frame(
                file_path=_get_depth_image_name(
                    image_path=image_number, depth_image_dir=depth_image_dir
                )
            )
            resized_seg_masks = resize_mask(
                seg_masks, height=depth_frame.shape[0], width=depth_frame.shape[1]
            )

            id_mapping_file = id_mapping_dir / f"{image_number}.json"

            seg_mappings = _match_mask_to_weight(
                seg_masks=resized_seg_masks,
                weights=weights,
                id_mapping_file=id_mapping_file,
                original_scale=_seg_masks.original_hw,
                rgb_transform_params=rgb_transform_params
            )

            for seg_mapping in seg_mappings:
                binary_mask = seg_mapping.seg_mask
                scale_x = depth_frame.shape[1] / binary_mask.shape[1]
                scale_y = depth_frame.shape[0] / binary_mask.shape[0]

                coordinates = np.column_stack(np.where(binary_mask == 1))

                for coord in coordinates:
                    rgb_y, rgb_x = coord
                    depth_x = int(rgb_x * scale_x)
                    depth_y = int(rgb_y * scale_y)

                    depth_value = depth_frame[depth_y, depth_x]
                    depth_values.append(depth_value)

    depth_masks = np.array(depth_values)
    mean = np.mean(depth_masks)
    std = np.std(depth_masks, ddof=1)
    print(f"Mean {mean} Standard Deviation {std}")


def _load_weights(weights_file) -> dict[str, float]:
    """
    Loads the weights file into a dictionary, with the key as the cattle ID
    """

    weights = {}

    with open(weights_file, mode="r") as file:
        csv_reader = csv.DictReader(file)

        for row in csv_reader:
            cattle_id = row["Name"]
            weight = float(row["Weight"])
            weights[cattle_id] = [weight]
    return weights


def _get_depth_image_name(image_path: str, depth_image_dir: Path):
    """
    Gets the depth image name.
    """
    return depth_image_dir / f"{image_path}.png"


def _match_mask_to_weight(
    seg_masks: list[np.ndarray],
    weights: dict[str, int],
    id_mapping_file: Path,
    rgb_transform_params: RgbTransformParams,
    original_scale: [int, int],
) -> list[SegMaskWeight]:
    """
    Finds the weight corresponding to each segmentation mask by using coordinates and IDs in the ID mapping file.
    """

    id_mappings = _load_id_mapping(id_mapping_file)
    mappings = []

    matched_ids = set()

    for mask in seg_masks:
        for id_mapping in id_mappings:
            cattle_id = id_mapping.id
            id_x, id_y = transform_id_coordinates(
                x_original=id_mapping.x,
                y_original=id_mapping.y,
                transform_params=rgb_transform_params,
            )
            map_y, map_x = _scale_coordinate(
                coordinate=(id_y, id_x),
                original_scale=(rgb_transform_params.target_height, rgb_transform_params.target_width),
                new_scale=mask.shape[:2],
            )

            if cattle_id not in matched_ids and mask[map_y][map_x] == 1:
                matched_ids.add(cattle_id)
                weight = weights[cattle_id]
                mappings.append(SegMaskWeight(seg_mask=mask, weight=weight))

    return mappings


def _load_id_mapping(file_path: Path) -> list[IdMapping]:
    """
    Loads the ID mapping file into a list of IdMapping.
    """

    json_dict = parse_json(file_path)

    id_mappings = []

    if json_dict.get("shapes") is not None and len(json_dict.get("shapes")) > 0:
        for shape in json_dict["shapes"]:
            points = shape["points"]
            if len(points) == 1:
                point = points[0]
                if len(point) != 2:
                    raise ValueError(f"Should be a x and y coordinate and not {points}")

                id_mappings.append(
                    IdMapping(
                        x=int(point[0]), y=int(point[1]), id=str(shape["group_id"])
                    )
                )

    return id_mappings


def _scale_coordinate(
    coordinate: tuple[int, int],
    original_scale: tuple[int, int],
    new_scale: tuple[int, int],
) -> tuple[int, int]:
    """
    Scales the coordinates.
    """

    scale_x = original_scale[1] / new_scale[1]
    scale_y = original_scale[0] / new_scale[0]

    scaled_x = int(coordinate[1] / scale_x)
    scaled_y = int(coordinate[0] / scale_y)
    return scaled_y, scaled_x


def _load_depth_frame(file_path: Union[str, Path]) -> np.ndarray:
    """
    Loads the depth image into a numpy array.
    """

    depth_image_array = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

    return depth_image_array


def _create_depth_masks(
    seg_mappings: list[SegMaskWeight], depth_frame: np.ndarray, scale_to_255: bool
) -> list[DepthMaskWeight]:
    """
    Creates the depth masks from the binary segmentation masks.
    """

    depth_masks = []

    for seg_mapping in seg_mappings:
        binary_mask = seg_mapping.seg_mask
        depth_mask = _create_depth_mask(binary_mask, depth_frame, scale_to_255)
        depth_masks.append(
            DepthMaskWeight(depth_mask=depth_mask, weight=seg_mapping.weight)
        )
        # show_mask(depth_mask)

    return depth_masks


def _create_depth_mask(binary_mask: np.ndarray, depth_frame: np.ndarray, scale_to_255: bool) -> np.ndarray:
    """
    Creates a depth mask from a binary segmentation mask.
    """

    scale_x = depth_frame.shape[1] / binary_mask.shape[1]
    scale_y = depth_frame.shape[0] / binary_mask.shape[0]

    output_mask = np.zeros_like(binary_mask, dtype=np.float32)
    coordinates = np.column_stack(np.where(binary_mask == 1))
    original_depth_values = []

    for coord in coordinates:
        rgb_y, rgb_x = coord
        depth_x = int(rgb_x * scale_x)
        depth_y = int(rgb_y * scale_y)

        original_depth_values.append(depth_frame[depth_y, depth_x])

    if scale_to_255:
        normalised_depth_values = _normalise_depth_values(depth_values=np.array(original_depth_values))
        normalised_depth_values = normalised_depth_values * 255
    else:
        normalised_depth_values = original_depth_values

    for i, coord in enumerate(coordinates):
        rgb_y, rgb_x = coord
        output_mask[rgb_y, rgb_x] = normalised_depth_values[i]

    # output_mask_normalized = cv2.normalize(output_mask, None, 0, 255, cv2.NORM_MINMAX)
    output_mask_normalized = output_mask
    scaled_mask = _scale_mask(output_mask_normalized)
    padded_image = _pad_image(scaled_mask)

    return padded_image


def _normalise_depth_values(depth_values: np.ndarray) -> np.ndarray:
    min_val = np.min(depth_values)
    max_val = np.max(depth_values)
    normalised_data = (depth_values - min_val) / (max_val - min_val)
    return normalised_data


def _scale_mask(mask: np.ndarray, dimension: int = SCALE_DIMENSION) -> np.ndarray:
    """
    Scales the mask to the given dimension but maintains the same aspect ratio.
    """
    height, width = mask.shape[:2]

    if height > width:
        scale_factor = dimension / height
    else:
        scale_factor = dimension / width

    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    resized_mask = cv2.resize(mask, (new_width, new_height))

    return resized_mask


def _pad_image(
    image: np.ndarray,
    height: int = DEPTH_MASK_DIMENSION[0],
    width: int = DEPTH_MASK_DIMENSION[1],
) -> np.ndarray:
    """
    Adds padding to the image so that it matches the specified height and width dimensions.
    """

    y, x = image.shape[:2]

    pad_y = (height - y) // 2
    pad_x = (width - x) // 2

    pad_top = pad_y
    pad_bottom = height - y - pad_top
    pad_left = pad_x
    pad_right = width - x - pad_left

    padded_image = np.pad(
        image,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=0,
    )

    return padded_image


def _save_depth_masks(depth_mappings: list[DepthMaskWeight], file_path: Path):
    """
    Saves depth masks to a JSON file called "depth_masks.json"
    """
    depth_masks = [obj.depth_mask for obj in depth_mappings]
    weights = [obj.weight for obj in depth_mappings]
    np.savez(file_path, depth_masks=depth_masks, weights=weights)

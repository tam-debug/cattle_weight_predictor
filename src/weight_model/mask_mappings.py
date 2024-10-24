"""
Create mappings between the masks and their corresponding weights.
Can apply segmentation masks to create new masks such as depth masks and RGB masks.
"""

import csv
import cv2
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
from typing import Union, Optional

import numpy as np

from constants.constants import SCALE_DIMENSION, DEPTH_MASK_DIMENSION
from segmentation_model.generate_masks import (
    ImageSegmentationMasks,
    generate_masks,
    load_seg_mask_tensors,
    load_prelabel_masks,
)
from segmentation_model.dataset_builder.resize_rgb import RgbTransformParams
from utils.utils import parse_json, resize_mask, show_mask

logger = logging.getLogger(__name__)


class MaskOutputType(Enum):
    DEPTH = 1
    SEGMENTATION = 2
    RGB = 3


@dataclass
class MaskWeight:
    """
    Contains a mask with its corresponding weight.
    """

    mask: np.ndarray
    weight: float


@dataclass
class IdMapping:
    """
    Contains the cattle id and corresponding coordinate contained by the cattle.
    """

    x: int
    y: int
    id: str


def load_dataset(file_paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads the masks and weight from the .npz file.

    :param file_paths: The file path to the .npz file.
    :return: The depth masks and weights in separate numpy arrays.
    """

    masks, weights = [], []

    for file_path in file_paths:
        data = np.load(file_path, allow_pickle=True)

        keys = data.keys()
        if len(keys) != 2:
            raise ValueError(f"Keys should be for masks and weights and not {keys}.")
        mask_key = [key for key in keys if key != "weights"][0]

        _masks = data[mask_key]
        _weights = data["weights"]

        if len(weights) == 0:
            weights = _weights
            masks.append(_masks)
        elif not np.array_equal(_weights, weights):
            raise ValueError("Weights are not the same for all datasets.")
        else:
            masks.append(_masks)

    if len(masks) > 1:
        if len(masks[0].shape) == 4:
            masks = np.concatenate(masks, axis=-1)
            masks = masks.transpose(0, 3, 1, 2)
        else:
            masks = np.stack(masks, axis=1)
    else:
        if len(masks[0].shape) == 4:
            masks = masks[0]
            masks = masks.transpose(0, 3, 1, 2)
        else:
            masks = np.array(masks)
            masks = masks.transpose(1, 0, 2, 3)

    return masks, np.array(weights)


def generate_mask_mappings(
    seg_masks: dict[int, ImageSegmentationMasks],
    weights_file: Path,
    mask_output_type: MaskOutputType,
    scale_to_255: bool = False,
    depth_image_dir: Path = None,
    rgb_image_dir: Path = None,
    save_dir: Path = None,
    id_mapping_dir: Path = None,
    rgb_transform_params: RgbTransformParams = None,
) -> list[MaskWeight]:
    """
    Generates the mappings between the masks and weights and applies segmentation masks to create new masks if needed.

    :param seg_masks: The segmentation masks.
    :param weights_file: The path to the CSV file that contains the weights.
    :param mask_output_type: The masks output type.
    :param scale_to_255: Whether to scale the mask values to be from 0 to 255.
    :param depth_image_dir: The directory to the depth images.
    :param rgb_image_dir: The directory to the RGB images.
    :param save_dir: The directory to save the mask mappings to. If this is not specified the mask mappings will not be
        saved.
    :param id_mapping_dir: The directory that contains the JSON files that have the image ID mapping points.
    :param rgb_transform_params: The transformation parameters that were used for aligning the RGB and depth images.
        Only include this if the images are the transformed RGB images.
    :return: The mask mappings.
    """

    weights = _load_weights(weights_file)

    mappings = []

    if id_mapping_dir:
        for image_number, _seg_masks in seg_masks.items():
            seg_masks = _seg_masks.masks.cpu().numpy()
            id_mapping_file = id_mapping_dir / f"{image_number}.json"

            seg_mappings = _match_mask_to_weight(
                seg_masks=seg_masks,
                weights=weights,
                id_mapping_file=id_mapping_file,
                original_scale=_seg_masks.original_hw,
                rgb_transform_params=rgb_transform_params,
            )

            if mask_output_type != MaskOutputType.SEGMENTATION:
                image = _load_image(
                    output_type=mask_output_type,
                    image_number=image_number,
                    image_directory=(
                        rgb_image_dir
                        if mask_output_type == MaskOutputType.RGB
                        else depth_image_dir
                    ),
                )

                _rgb_mappings = _apply_seg_masks(
                    seg_mappings=seg_mappings, image=image, scale_to_255=scale_to_255
                )

                mappings.extend(_rgb_mappings)
            else:
                for seg_mapping in seg_mappings:
                    seg_mapping.mask = _pad_image(seg_mapping.mask)
                mappings.extend(seg_mappings)

    else:
        for image_number, _seg_masks in seg_masks.items():
            weight = weights[str(image_number)]
            binary_mask = _seg_masks.masks[0].cpu().numpy()

            if mask_output_type == MaskOutputType.SEGMENTATION:
                mask = binary_mask
            else:
                image = _load_image(
                    output_type=mask_output_type,
                    image_number=image_number,
                    image_directory=(
                        rgb_image_dir
                        if mask_output_type == MaskOutputType.RGB
                        else depth_image_dir
                    ),
                )

                mask = _apply_seg_mask(
                    binary_mask=binary_mask, image=image, scale_to_255=scale_to_255
                )

            mappings.append(MaskWeight(mask=mask, weight=weight))

    if save_dir:
        _save_masks(
            mappings=mappings, file_path=save_dir, mask_key=f"{mask_output_type}_masks"
        )

    return mappings


def print_masks_mean_and_stddev(
    seg_masks: dict[int, ImageSegmentationMasks],
    rgb_image_dir: Path,
    depth_image_dir: Path,
    mask_output_type: MaskOutputType,
    weights_file: Path,
    rgb_transform_params: RgbTransformParams = None,
    id_mapping_dir: Path = None,
):
    """
    Prints the mean and standard deviation of the values in the images that are in the segmentation masks.

    :param seg_masks: The segmentation masks.
    :param weights_file: The path to the CSV file that contains the weights.
    :param mask_output_type: The masks output type.
    :param depth_image_dir: The directory to the depth images.
    :param rgb_image_dir: The directory to the RGB images.
    :param id_mapping_dir: The directory that contains the JSON files that have the image ID mapping points.
    :param rgb_transform_params: The transformation parameters that were used for aligning the RGB and depth images.
        Only include this if the images are the transformed RGB images.
    """

    weights = _load_weights(weights_file)

    values = []
    if id_mapping_dir:
        for image_number, _seg_masks in seg_masks.items():
            seg_masks = _seg_masks.masks.cpu().numpy()
            id_mapping_file = id_mapping_dir / f"{image_number}.json"

            image = _load_image(
                output_type=mask_output_type,
                image_number=image_number,
                image_directory=(
                    rgb_image_dir
                    if mask_output_type == MaskOutputType.RGB
                    else depth_image_dir
                ),
            )

            seg_mappings = _match_mask_to_weight(
                seg_masks=seg_masks,
                weights=weights,
                id_mapping_file=id_mapping_file,
                original_scale=_seg_masks.original_hw,
                rgb_transform_params=rgb_transform_params,
            )
            for seg_mapping in seg_mappings:
                _values = _get_image_values_in_seg_mask(
                    binary_mask=seg_mapping.mask, image=image
                )
                values.extend(_values)

    else:
        for image_number, _seg_masks in seg_masks.items():
            binary_mask = _seg_masks.masks[0].cpu().numpy()

            image = _load_image(
                output_type=mask_output_type,
                image_number=image_number,
                image_directory=(
                    rgb_image_dir
                    if mask_output_type == MaskOutputType.RGB
                    else depth_image_dir
                ),
            )
            _values = _get_image_values_in_seg_mask(
                binary_mask=binary_mask, image=image
            )
            values.extend(_values)

    mean_rgb = np.mean(values, axis=0)
    std_rgb = np.std(values, axis=0)

    print(f"Mean {mean_rgb} Standard Deviation {std_rgb}")


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
            weights[cattle_id] = weight
    return weights


def _load_image(
    output_type: str, image_number: int, image_directory: Path
) -> np.ndarray:
    """
    Loads the depth or RGB image.
    """
    file_path = _get_image_name(image_path=image_number, image_dir=image_directory)

    if output_type == MaskOutputType.DEPTH:
        image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    else:
        image = cv2.imread(file_path)

    return image


def _get_image_name(image_path: Union[str, int], image_dir: Path):
    """
    Gets the depth or RGB image name.
    """
    return image_dir / f"{image_path}.png"


def _match_mask_to_weight(
    seg_masks: np.ndarray,
    weights: dict[str, float],
    id_mapping_file: Path,
    original_scale: [int, int],
    rgb_transform_params: Optional[RgbTransformParams] = None,
) -> list[MaskWeight]:
    """
    Finds the weight corresponding to each segmentation mask by using coordinates and IDs in the ID mapping file.
    """

    id_mappings = _load_id_mapping(id_mapping_file)
    mappings = []

    matched_ids = set()

    for mask in seg_masks:
        for id_mapping in id_mappings:
            cattle_id = id_mapping.id
            if rgb_transform_params:
                id_x, id_y = _transform_id_coordinates(
                    x_original=id_mapping.x,
                    y_original=id_mapping.y,
                    transform_params=rgb_transform_params,
                )
                orig_height, orig_width = (
                    rgb_transform_params.target_height,
                    rgb_transform_params.target_width,
                )
            else:
                id_x, id_y = id_mapping.x, id_mapping.y
                orig_height, orig_width = original_scale[0], original_scale[1]

            map_y, map_x = _scale_coordinate(
                coordinate=(id_y, id_x),
                original_scale=(orig_height, orig_width),
                new_scale=mask.shape[:2],
            )

            if cattle_id not in matched_ids and mask[map_y][map_x] == 1:
                matched_ids.add(cattle_id)
                weight = weights[cattle_id]
                mappings.append(MaskWeight(mask=mask, weight=weight))

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


def _transform_id_coordinates(
    x_original: int, y_original: int, transform_params: RgbTransformParams
) -> tuple[int, int]:
    """
    Transforms the coordinates, so they can be used for the RGB aligned image.
    """
    # Step 1: Apply padding (shift y coordinate by 'top_pad')
    x_padded = x_original
    y_padded = y_original + transform_params.top_pad

    # Step 2: Apply cropping (shift x coordinate by 'left_crop')
    x_cropped = x_padded - transform_params.left_crop
    y_cropped = y_padded

    # Intermediate image size after padding and cropping
    intermediate_width = (
        transform_params.original_width
        - transform_params.left_crop
        - transform_params.right_crop
    )
    intermediate_height = (
        transform_params.original_height
        + transform_params.top_pad
        + transform_params.bottom_pad
    )

    # Step 3: Apply resizing to (512, 424)
    x_resized = x_cropped * (transform_params.target_width / intermediate_width)
    y_resized = y_cropped * (transform_params.target_height / intermediate_height)

    return int(x_resized), int(y_resized)


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


def _apply_seg_masks(
    seg_mappings: list[MaskWeight], image: np.ndarray, scale_to_255: bool = False
) -> list[MaskWeight]:
    """
    Creates new masks by applying the binary segmentation masks on the image.
    """

    masks = []

    for seg_mapping in seg_mappings:
        binary_mask = seg_mapping.mask
        new_mask = _apply_seg_mask(binary_mask, image, scale_to_255=scale_to_255)
        masks.append(MaskWeight(mask=new_mask, weight=seg_mapping.weight))
        # show_mask(depth_mask)

    return masks


def _apply_seg_mask(
    binary_mask: np.ndarray, image: np.ndarray, scale_to_255: bool = False
) -> np.ndarray:
    """
    Creates a new mask by applying the binary segmentation mask on the image.
    """

    mask_values = _get_image_values_in_seg_mask(binary_mask=binary_mask, image=image)

    if scale_to_255:
        mask_values = _normalise_values(values=np.array(mask_values)) * 255

    if len(image.shape) == 3 and image.shape[2] == 3:
        output_mask = np.zeros((binary_mask.shape[0], binary_mask.shape[1], 3), dtype=np.float32)
    else:
        output_mask = np.zeros_like(binary_mask, dtype=np.float32)
    coordinates = np.column_stack(np.where(binary_mask == 1))

    for i, coord in enumerate(coordinates):
        seg_y, seg_x = coord
        output_mask[seg_y, seg_x] = mask_values[i]

    scaled_mask = _scale_mask(output_mask)
    padded_image = _pad_image(scaled_mask)

    return padded_image


def _get_image_values_in_seg_mask(binary_mask: np.ndarray, image: np.ndarray) -> list:
    """
    Gets the image values that are in the segmentation mask.
    """
    values = []
    scale_x = image.shape[1] / binary_mask.shape[1]
    scale_y = image.shape[0] / binary_mask.shape[0]

    coordinates = np.column_stack(np.where(binary_mask == 1))

    for coord in coordinates:
        seg_y, seg_x = coord
        x = int(seg_x * scale_x)
        y = int(seg_y * scale_y)

        values.append(image[y, x])
    return values


def _normalise_values(values: np.ndarray) -> np.ndarray:
    """
    Normalises the values, so they are between 0 and 1.
    """
    min_val = np.min(values)
    max_val = np.max(values)
    normalised_data = (values - min_val) / (max_val - min_val)
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

    if len(image.shape) == 3 and image.shape[2] == 3:
        padded_image = np.pad(
            image,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode='constant',
            constant_values=0
        )
    else:
        padded_image = np.pad(
            image,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=0,
        )

    return padded_image


def _save_masks(mappings: list[MaskWeight], file_path: Path, mask_key: str):
    """
    Saves depth masks to a JSON file called "depth_masks.json"
    """
    masks = [mapping.mask for mapping in mappings]
    weights = [seg_mapping.weight for seg_mapping in mappings]
    params = {mask_key: masks, "weights": weights}
    np.savez(file_path, **params)

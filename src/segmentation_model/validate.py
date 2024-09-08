"""
Runs the model on an input and generates the IOU for the predictions.
"""

from pathlib import Path

from ultralytics import YOLO

from segmentation_model.iou import PredictionResult, calculate_ious


def validate(
    model: YOLO, project: Path, name: str, input_directory: Path, labels_directory: Path
) -> list[float]:
    """
    Validates the model against the input, with IOU as the main metric.

    :param model: The model to run validation on.
    :param project: The directory the validation results will be saved.
    :param name: The name of the folder that will store the validation results.
    :param labels_directory: The directory where the labels are stored.
    :param input_directory: The directory to run the model on.
    :return: The IOUs from the validation.
    """
    prediction_masks = []
    conf_threshold = 0.5

    results = model.predict(
        source=input_directory, project=project, save=True, imgsz=640, name=name
    )

    for result in results:
        image_id = Path(result.path).stem
        bounding_box_conf = result.boxes.conf
        result_masks = result.masks
        masks = []
        if result_masks:
            for i in range(len(bounding_box_conf)):
                if bounding_box_conf[i] > conf_threshold:
                    masks.append(result_masks.data[i].cpu().numpy())

        prediction_masks.append(
            PredictionResult(
                image_id=image_id,
                masks=masks,
                mask_size=_scale_coordinate_same_aspect_ratio(
                    height=result.orig_shape[0],
                    width=result.orig_shape[1],
                    dimension=640,
                ),
            )
        )

    ious = calculate_ious(
        labels_directory=labels_directory, prediction_results=prediction_masks
    )
    return ious


def _scale_coordinate_same_aspect_ratio(
    height, width, dimension: int = 640
) -> tuple[int, int]:
    """
    Scales the coordinate to the given dimension.
    """
    if height > width:
        scale_factor = dimension / height
    else:
        scale_factor = dimension / width

    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    return new_height, new_width

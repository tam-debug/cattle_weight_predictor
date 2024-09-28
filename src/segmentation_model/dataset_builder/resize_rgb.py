from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

@dataclass
class RgbTransformParams:
    top_pad: int
    bottom_pad: int
    left_crop: int
    right_crop: int
    target_width: int
    target_height: int
    original_width: int
    original_height: int

def prepare_rgb(
    input_dir: Path,
    save_dir: Path,
    transform_params: RgbTransformParams,
    start_num: int,
    end_num: int,
):

    for i in range(start_num, end_num + 1):

        image_path = input_dir / f"{i}.png"
        save_path = save_dir / f"{i}.png"

        image = Image.open(image_path)

        new_image = ImageOps.expand(
            image, border=(0, transform_params.top_pad, 0, transform_params.bottom_pad), fill=(0, 0, 0)
        )  # Black padding (change 'fill' if needed)

        width, height = new_image.size
        cropped_image = new_image.crop((transform_params.left_crop, 0, width - transform_params.right_crop, height))

        resized_image = cropped_image.resize((transform_params.target_width, transform_params.target_height))

        resized_image.save(save_path)


def transform_id_coordinates(
    x_original: int, y_original: int, transform_params: RgbTransformParams
) -> tuple[int, int]:
    # Step 1: Apply padding (shift y coordinate by 'top_pad')
    x_padded = x_original
    y_padded = y_original + transform_params.top_pad

    # Step 2: Apply cropping (shift x coordinate by 'left_crop')
    x_cropped = x_padded - transform_params.left_crop
    y_cropped = y_padded

    # Intermediate image size after padding and cropping
    intermediate_width = transform_params.original_width - transform_params.left_crop - transform_params.right_crop
    intermediate_height = transform_params.original_height + transform_params.top_pad + transform_params.bottom_pad

    # Step 3: Apply resizing to (512, 424)
    x_resized = x_cropped * (transform_params.target_width / intermediate_width)
    y_resized = y_cropped * (transform_params.target_height / intermediate_height)

    return int(x_resized), int(y_resized)

from abc import abstractmethod, ABC
from enum import Enum
import math
import os
from pathlib import Path
import random
import shutil
from typing import Optional
import zipfile

from PIL import Image


class ValidationSelectionType(Enum):
    FIRST = 1
    LAST = 2
    RANDOM = 3


YAML_FILENAME = "custom_dataset"
TRAINING_DIR = "train"
TEST_DIR = "test"
IMAGES_DIR = "images"
LABELS_DIR = "labels"
DEFAULT_VALIDATION_PORTION = 0.2
DEFAULT_VALIDATION_SELECTION = ValidationSelectionType.RANDOM
DEFAULT_ZIP_FILENAME = "datasets"


class DatasetEntry:
    """
    The dataset entry which contains the file path of where the image is currently located.
    """

    def __init__(self, image_path: Path, label_path: Path):
        self.image_path = image_path
        self.label_path = label_path

    def add_to_dataset(
        self,
        destination_label_dir: Path,
        destination_image_dir: Path,
        image_resize: tuple = None,
    ) -> None:
        """
        Adds this dataset entry to the custom dataset.

        This involves copying the label file and image to the destination directory.
        :param destination_label_dir: The destination path the label file will be copied to.
        :param destination_image_dir: The directory that the image file will be copied to.
        :param image_resize: The image size to resize to, using padding. If this is (None, None), the image will not
            be resized and padded.
        """
        if os.path.exists(self.label_path):
            shutil.copy(self.label_path, destination_label_dir / self.label_path.name)
        else:
            print(f"Adding empty label file for {self.label_path.name}")
            with open(destination_label_dir / self.label_path.name, "w"):
                pass
        if image_resize is None:
            shutil.copy(self.image_path, destination_image_dir / self.image_path.name)
        else:
            self._pad_and_resize_image(destination_image_dir, image_resize)

    def _pad_and_resize_image(
        self, destination_dir: Path, image_resize: tuple[int, int]
    ) -> None:
        img = Image.open(self.image_path)
        img_width, img_height = img.size

        add_width = 0
        add_height = 0
        if img_width > img_height:
            add_height = img_width - img_height
        elif img_height > img_width:
            add_width = img_height - img_width

        new_width, new_height = img_width + add_width, img_height + add_height
        new_img = Image.new(img.mode, (new_width, new_height), (0, 0, 0))
        new_img.paste(img, (int(add_width / 2), int(add_height / 2)))

        new_img = new_img.convert("RGB")
        new_img = new_img.resize(image_resize)
        new_img.save(destination_dir / self.image_path.name)


class ValidationSet(ABC):
    @abstractmethod
    def is_part_of_validation(self, image_number: int) -> bool:
        pass


class DatasetSplit(ValidationSet):
    def __init__(
        self,
        end_number: int,
        proportion: float = DEFAULT_VALIDATION_PORTION,
        start_number: int = 0,
    ):
        self.end_number = end_number
        self.start_number = start_number
        self.split = math.floor(proportion * (end_number - start_number))


class FirstSplit(DatasetSplit):
    def is_part_of_validation(self, image_number: int) -> bool:
        return image_number < self.split + self.start_number


class LastSplit(DatasetSplit):
    def is_part_of_validation(self, image_number) -> bool:
        return image_number > self.end_number - self.split


class RandomSplit(DatasetSplit):
    def __init__(
        self,
        end_number: int,
        proportion: float = DEFAULT_VALIDATION_PORTION,
        start_number: int = 0,
    ):
        super(RandomSplit, self).__init__(end_number=end_number, proportion=proportion)
        self.validation_set = self._get_random_numbers_proportion()

    def is_part_of_validation(self, image_number: int) -> bool:
        return image_number in self.validation_set

    def _get_random_numbers_proportion(self) -> set:
        random_numbers = set()
        while len(random_numbers) < self.split:
            number = random.randint(self.start_number, self.end_number)
            if number not in random_numbers:
                random_numbers.add(number)
        return random_numbers


class ValidationSetSplit(ValidationSet):
    def __init__(self, validation_set: set):
        self.validation_set = validation_set

    def is_part_of_validation(self, image_number: int) -> bool:
        return image_number in self.validation_set


class CustomDatasetGenerator:
    """
    Used for creating the custom dataset.
    """

    def __init__(
        self,
        data_dir: Path,
        labels_dir: Path,
        end_number: int,
        image_formats: list[str],
        validation_set: ValidationSet,
        start_number: int = 0,
        image_resize: tuple = None,
        zipfile_name: str = DEFAULT_ZIP_FILENAME,
    ):
        self.data_dir = data_dir
        self.labels_dir = labels_dir
        self.start_number = start_number
        self.end_number = end_number
        self.image_formats = image_formats
        self.dataset_split = validation_set

        self.train_dir = Path(TRAINING_DIR)
        self.test_dir = Path(TEST_DIR)
        self.images_dir = Path(IMAGES_DIR)
        self.dest_labels_dir = Path(LABELS_DIR)
        self.yaml_filename = Path(YAML_FILENAME) / ".yaml"
        self.image_resize = image_resize
        self.zipfile_name = zipfile_name

    def _get_image_path(
        self, image_path_no_ext: Path, image_formats: list
    ) -> Optional[Path]:
        for file_ext in image_formats:
            path_to_test = image_path_no_ext.with_suffix(f".{file_ext}")
            if os.path.exists(path_to_test):
                return path_to_test
        return None

    def generate_dataset(self) -> None:
        """
        Creates the custom dataset.
        """
        self._create_directories()
        for number in range(self.start_number, self.end_number + 1):
            image_id = f"{number:05d}"
            image_path = self._get_image_path(
                self.data_dir / image_id, self.image_formats
            )

            if image_path is not None:
                entry = DatasetEntry(
                    image_path=image_path,
                    label_path=self.labels_dir / f"{image_id}.txt",
                )
                is_validation = self.dataset_split.is_part_of_validation(number)
                parent_dir = self.test_dir if is_validation else self.train_dir

                entry.add_to_dataset(
                    destination_label_dir=parent_dir / self.dest_labels_dir,
                    destination_image_dir=parent_dir / self.images_dir,
                    image_resize=self.image_resize,
                )
            else:
                print(f"Warning: {number} does not exist")

        self._create_dataset_yaml()
        self._create_zip()

    def _create_directories(self) -> None:
        """
        Creates the directories that will store the images and their labels.
        """
        directories_to_create = [
            self.train_dir / self.images_dir,
            self.train_dir / self.dest_labels_dir,
            self.test_dir / self.images_dir,
            self.test_dir / self.dest_labels_dir,
        ]
        for directory in directories_to_create:
            if not os.path.exists(directory):
                os.makedirs(directory)

    def _create_dataset_yaml(self) -> None:
        """
        Creates a basic yaml for the dataset and model configuration.
        Has one class and specifies the training data path.
        """
        cattle_head_folder = "_".join(self.zipfile_name.split("_")[:2])
        parent_folder = (
            f"../../../Data/{cattle_head_folder}/{Path(self.zipfile_name).stem}"
        )

        with open(f"{YAML_FILENAME}.yaml", "w") as yaml_file:
            yaml_file.write(
                f"train: {parent_folder}/{(self.train_dir / self.images_dir).as_posix()}\n"
            )
            yaml_file.write(
                f"val: {parent_folder}/{(self.test_dir / self.images_dir).as_posix()}\n"
            )
            yaml_file.write("nc: 1")

    def _create_zip(self):
        with zipfile.ZipFile(f"{self.zipfile_name}", "w", zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(f"{YAML_FILENAME}.yaml")
            self._zip_directory(zipf, self.train_dir)
            self._zip_directory(zipf, self.test_dir)

    def _zip_directory(self, zip_file, directory):
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                zip_file.write(
                    file_path, os.path.relpath(file_path, os.path.join(directory, ".."))
                )


def generate_custom_dataset(
    end_number: int,
    data_dir: str,
    labels_dir: str,
    image_formats: list[str],
    proportion: float = DEFAULT_VALIDATION_PORTION,
    validation_selection_type: ValidationSelectionType = DEFAULT_VALIDATION_SELECTION,
    start_number: int = 0,
    image_resize: tuple = None,
    validation_set: set = None,
    zipfile_name: str = DEFAULT_ZIP_FILENAME,
):
    """
    Creates the custom dataset.

    :param end_number: The number of the last image.
    :param data_dir: The directory where the images are currently stored.
    :param labels_dir: The directory where the label files are currently stored.
    :param image_formats: The file format suffixes of the images e.g. jpg, jpeg, png
    :param proportion: The fraction of dataset to be used for validation. E.g. 0.2
    :param validation_selection_type: The method used to determine what data is selected for validation.
    :param start_number: The number of the first image.
    :param image_resize: The dimensions to resize the images to. If None, the images will not be resized.
    :param validation_set: Used for specifying which image ids are in the validation set, if the validation set is
        already predetermined.
    :param zipfile_name: The name of the zipfile to save the dataset to.

    :raises ValueError: If the proportion is not between 0 and 1.
    """
    if validation_set is not None:
        if len(validation_set) >= end_number - start_number:
            raise ValueError(
                f"The number of elements in the validation set should not exceed the number of elements in the dataset"
            )
        validation_set = ValidationSetSplit(validation_set=validation_set)
    else:
        if proportion <= 0 or proportion >= 1:
            raise ValueError(f"The proportion {proportion} is not between 0 and 1.")

        validation_selection_map = {
            ValidationSelectionType.FIRST: FirstSplit,
            ValidationSelectionType.LAST: LastSplit,
            ValidationSelectionType.RANDOM: RandomSplit,
        }
        validation_set = validation_selection_map[validation_selection_type](
            end_number=end_number, proportion=proportion, start_number=start_number
        )
    custom_dataset = CustomDatasetGenerator(
        data_dir=Path(data_dir),
        labels_dir=Path(labels_dir),
        end_number=end_number,
        image_formats=image_formats,
        validation_set=validation_set,
        start_number=start_number,
        image_resize=image_resize,
        zipfile_name=zipfile_name,
    )

    custom_dataset.generate_dataset()

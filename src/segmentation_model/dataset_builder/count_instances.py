"""
Gets the instances information from the dataset.
"""

from dataclasses import dataclass
import os
from pathlib import Path

import pandas as pd


@dataclass
class DatasetInfo:
    directory_path: Path
    image_count: int
    instance_count: int
    blank_count: int


def generate_k_fold_dataset_info(
    train_directory: Path, test_directory, output_file_path: Path
) -> pd.DataFrame:
    """
    Generates the dataset information for a k_fold dataset to a file.
    :param train_directory:
    :param test_directory:
    :param output_file_path:
    :return: The dataset information in a pandas DataFrame.
    """
    split_dirs = [path for path in train_directory.iterdir() if path.is_dir()]
    df = pd.DataFrame(
        columns=["dataset_component", "image_count", "instance_count", "blank_count"]
    )

    for i, split_dir in enumerate(split_dirs, start=1):
        for sub_dir in ["train", "val"]:
            df = _add_info_row(
                df=df,
                path=split_dir / sub_dir,
                dataset_component=f"{split_dir.stem}_{sub_dir}",
            )

    df = _add_info_row(
        df=df, path=test_directory, dataset_component=test_directory.stem
    )

    df.to_csv(output_file_path, index=False)
    return df


def get_info_per_file(directory) -> DatasetInfo:
    """
    Gets the dataset information for the given directory.

    :param directory: The directory to get the information from.
    :return: The dataset information.
    """
    label_files = _get_label_files(directory)
    instance_count = 0
    blank_count = 0

    for label_file in label_files:
        if _is_file_empty(label_file):
            blank_count += 1
        else:
            instance_count += _count_non_empty_lines(label_file)

    return DatasetInfo(
        directory_path=directory,
        image_count=len(label_files),
        instance_count=instance_count,
        blank_count=blank_count,
    )


def _add_info_row(df: pd.DataFrame, path: Path, dataset_component: str) -> pd.DataFrame:
    """
    Adds the dataset information for a given path to the dataframe.
    """
    info = get_info_per_file(path)

    row = {
        "dataset_component": dataset_component,
        "image_count": info.image_count,
        "instance_count": info.instance_count,
        "blank_count": info.blank_count,
    }

    df = df._append(row, ignore_index=True)
    return df


def _get_label_files(file_path: Path) -> list[Path]:
    """
    Gets all the text files in the directory and returns their paths in a list.
    """
    label_files = []
    for root, dirs, files in os.walk(file_path):
        for file in files:
            if file.lower().endswith(".txt"):
                label_files.append(Path(os.path.join(root, file)))

    return label_files


def _is_file_empty(file_path: Path) -> bool:
    """Checks if  a file is empty."""
    with open(file_path, "r") as file:
        content = file.read().strip()
    return not content


def _count_non_empty_lines(file_path: Path) -> int:
    """Counts the number of non-empty lines."""

    with open(file_path, "r") as file:
        non_empty_lines = sum(1 for line in file if line.strip())
    return non_empty_lines

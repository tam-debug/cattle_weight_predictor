"""
Methods for configure and creating dataset splits.
"""

from collections import Counter
import datetime
import logging
from pathlib import Path
import random
import shutil
import yaml

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from constants.constants import K_FOLD_DATASET_INFO_FILENAME

logger = logging.getLogger(__name__)


def generate_train_test_split(numbers: range, test_proportion: float, file_path: Path):
    """
    Generates the configuration for a train and test split.
    :param numbers: The numbers to split.
    :param test_proportion: The proportion of the test split.
    :param file_path: The file path to save the configuration to.
    """

    test_size = int(test_proportion * len(list(numbers)))

    test_set = random.sample(numbers, test_size)
    test_set_array = np.array(test_set)

    np.save(file_path, test_set_array)


def generate_k_fold_configuration(n_splits: int, data, file_path: Path):
    """
    Generates the k-folds and save them to a .npy

    :param n_splits: The number of splits to perform.
    :param data: The data to split.
    :param file_path: The file path to save the file.
    """
    k_fold = KFold(n_splits=n_splits, shuffle=True, random_state=91231088)

    if logger.root.level <= logging.DEBUG:
        for fold, (train_index, test_index) in enumerate(k_fold.split(data)):
            logger.debug(f"Fold {fold + 1}")
            logger.debug(f"Train indices: {train_index}")
            logger.debug(f"Test indices: {test_index}\n")
    folds = {}

    i = 0
    for train_index, test_index in k_fold.split(data):
        folds[f"list_{i}"] = train_index.tolist()
        i += 1
        folds[f"list_{i}"] = test_index.tolist()
        i += 1

    if file_path.suffix != ".npz":
        raise ValueError(f"Invalid .npz suffix {file_path}")

    np.savez(file_path, **folds)


def load_k_fold(file_path) -> list[tuple[np.ndarray]]:
    """
    Load the k fold configuration.
    :param file_path: The file that contains the k fold configuration.
    :return: The k fold configuration.
    """

    if file_path.suffix != ".npz":
        raise ValueError(f"Invalid .npz suffix {file_path}")

    loaded_folds = np.load(file_path, allow_pickle=True)
    loaded_folds_list = [loaded_folds[key] for key in loaded_folds.keys()]

    folds = []
    for i in range(0, len(loaded_folds_list), 2):
        folds.append((loaded_folds_list[i], loaded_folds_list[i + 1]))

    return folds


def create_k_fold_datasets(
    image_path: Path, label_path: Path, k_folds_path: Path, output_dir: str
):
    """
    Splits the dataset into k folds.

    :param image_path: The directory that contains the image files.
    :param label_path: The directory that contains the label files.
    :param k_folds_path: The path to the k fold configuration.
    :param output_dir: The directory that will contain the resultant dataset.
    """

    labels = sorted(label_path.rglob("*.txt"))
    cls_index = [0]
    index = [label_path.stem for label_path in labels]
    labels_df = pd.DataFrame([], columns=cls_index, index=index)

    for label in labels:
        label_counter = Counter()

        with open(label, "r") as lf:
            lines = lf.readlines()

        for line in lines:
            # classes for YOLO label uses integer at first position of each line
            if line.strip() != "":
                label_counter[int(line.split(" ")[0])] += 1

        labels_df.loc[label.stem] = label_counter

    labels_df = labels_df.fillna(0)

    k_split = 5
    k_folds = load_k_fold(k_folds_path)
    folds = [f"split_{n}" for n in range(1, k_split + 1)]
    folds_df = pd.DataFrame(index=index, columns=folds)

    for idx, (train, val) in enumerate(k_folds, start=1):
        folds_df[f"split_{idx}"].loc[labels_df.iloc[train].index] = "train"
        folds_df[f"split_{idx}"].loc[labels_df.iloc[val].index] = "val"

    supported_extensions = [".jpg", ".jpeg", ".png"]
    images = []

    for ext in supported_extensions:
        images.extend(sorted(image_path.rglob(f"*{ext}")))

    save_path = Path(f"{datetime.date.today().isoformat()}_{k_split}_fold_cross_val")
    save_path.mkdir(parents=True, exist_ok=True)
    ds_yamls = []

    for i, split in enumerate(folds_df.columns):
        split_dir = save_path / split
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
        (split_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "val" / "labels").mkdir(parents=True, exist_ok=True)

        dataset_yaml = split_dir / f"{split}_dataset.yaml"
        ds_yamls.append(dataset_yaml)

        with open(dataset_yaml, "w") as ds_y:
            yaml.safe_dump(
                {
                    "nc": 1,
                    "train": f"{output_dir}/train/split_{i+1}/train",
                    "val": f"{output_dir}/train/split_{i+1}/val",
                    "test": f"{output_dir}/train/test",
                }, ds_y
            )

    for image, label in zip(images, labels):
        for split, k_split in folds_df.loc[image.stem].items():
            img_to_path = save_path / split / k_split / "images"
            label_to_path = save_path / split / k_split / "labels"

            # Copy image and label files to new directory (SamefileError if file already exists)
            shutil.copy(image, img_to_path / image.name)
            shutil.copy(label, label_to_path / label.name)

    folds_df.to_csv(save_path / K_FOLD_DATASET_INFO_FILENAME)

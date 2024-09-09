"""
Methods for handling the k-fold configuration.
"""

from dataclasses import dataclass
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

from weight_model.depth_masks import load_dataset

logger = logging.getLogger(__name__)


@dataclass
class DatasetFold:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def generate_k_fold_configuration(n_splits: int, data, json_file_path: Path):
    """
    Generates the k-folds and save them to a json file.

    :param n_splits: The number of splits to perform.
    :param data: The data to split.
    :param json_file_path: The file path to save the json file.
    """
    k_fold = KFold(n_splits=n_splits, shuffle=True, random_state=91231088)

    if logger.root.level <= logging.DEBUG:
        for fold, (train_index, test_index) in enumerate(k_fold.split(data)):
            logger.debug(f"Fold {fold + 1}")
            logger.debug(f"Train indices: {train_index}")
            logger.debug(f"Test indices: {test_index}\n")

    folds = [
        (train_idx.tolist(), test_idx.tolist())
        for train_idx, test_idx in k_fold.split(data)
    ]
    if json_file_path.suffix != ".json":
        raise ValueError(f"Invalid json_file_path suffix {json_file_path}")

    with open(json_file_path, "w") as file:
        json.dump(folds, file)


def generate_dataset_folds(
    folds_json_path: Path, data_paths: list[Path], output_json_file: Path = None
) -> list[DatasetFold]:
    """
    Generates dataset folds, which can optionally saved in a JSON file.

    :param folds_json_path: Path to the file that contains the configuration of the folds.
    :param data_paths: The paths JSON files that store the data.
    :param output_json_file: The output JSON file to store the dataset folds.
    :return: The dataset folds.
    """
    folds = _load_folds_configuration(json_file_path=folds_json_path)

    depth_masks, weights = load_dataset(data_paths)

    dataset = _match_data_to_folds(folds=folds, X=depth_masks, y=weights)
    if output_json_file:
        _save_dataset_folds(dataset_folds=dataset, file_path=output_json_file)

    return dataset


def load_dataset_folds(file_path: Path) -> list[DatasetFold]:
    """
    Load dataset folds from a JSON file.
    :param file_path: The path to the JSON file that contains the dataset folds.
    :return: The dataset folds.
    """
    with open(file_path, "r") as json_file:
        folds_data = json.load(json_file)

    folds = []
    for fold_data in folds_data:
        folds.append(
            DatasetFold(
                X_train=np.array(fold_data["X_train"]),
                y_train=np.array(fold_data["y_train"]),
                X_test=np.array(fold_data["X_test"]),
                y_test=np.array(fold_data["y_test"]),
            )
        )

    for i, fold in enumerate(folds):
        print(f"Fold {i + 1}:")
        print(f"  X_train shape: {fold.X_train.shape}")
        print(f"  y_train shape: {fold.y_train.shape}")
        print(f"  X_test shape: {fold.X_test.shape}")
        print(f"  y_test shape: {fold.y_test.shape}")

    return folds


def _load_folds_configuration(
    json_file_path: Path,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Loads the folds from a json file.

    :param json_file_path: The json file that stores the folds.
    :return: The folds as tuples (train, test) in a list.
    """
    with open(json_file_path, "r") as file:
        loaded_folds = json.load(file)

    loaded_folds = [
        (np.array(train_idx), np.array(test_idx))
        for train_idx, test_idx in loaded_folds
    ]

    if logger.root.level <= logging.DEBUG:
        for train_idx, test_idx in loaded_folds:
            logger.debug("Train indices:", train_idx, "Test indices:", test_idx)

    return loaded_folds


def _match_data_to_folds(
    folds: list[tuple[np.ndarray, np.ndarray]], X: np.ndarray, y: np.ndarray
) -> list[DatasetFold]:
    """
    Arrange the data into the folds.
    """
    dataset_folds = []

    for train_indices, test_indices in folds:
        train_masks = X[:, train_indices]
        train_weights = y[train_indices]

        test_masks = X[:, test_indices]
        test_weights = y[test_indices]

        dataset_folds.append(
            DatasetFold(
                X_train=train_masks,
                y_train=train_weights,
                X_test=test_masks,
                y_test=test_weights,
            )
        )

    if logger.root.level <= logging.DEBUG:
        for dataset_fold in dataset_folds:
            logger.debug("Train")
            logger.debug(dataset_fold.X_train)
            logger.debug(dataset_fold.y_train)

            logger.debug("Test")
            logger.debug(dataset_fold.X_test)
            logger.debug(dataset_fold.y_test)

    return dataset_folds


def _save_dataset_folds(dataset_folds: list[DatasetFold], file_path: Path):
    """
    Save the dataset folds to a JSON file.
    """

    if file_path.suffix != ".json":
        raise ValueError(f"Invalid JSON file suffix {file_path}")

    folds_serializable = []
    for fold in dataset_folds:
        fold_data = {
            "X_train": fold.X_train.tolist(),
            "y_train": fold.y_train.tolist(),
            "X_test": fold.X_test.tolist(),
            "y_test": fold.y_test.tolist(),
        }
        folds_serializable.append(fold_data)

    with open(file_path, "w") as json_file:
        json.dump(folds_serializable, json_file, indent=4)

"""
Methods for handling the k-fold configuration.
"""

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)


def generate_k_fold(n_splits: int, data, json_file_path: Path):
    """
    Generates the k-folds and save them to a json file.

    :param n_splits: The number of splits to perform.
    :param data: The data to split.
    :param json_file_path: The file path to save the json file.
    """
    k_fold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

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


def load_folds(json_file_path: Path) -> list:
    """
    Loads the folds from a json file.

    :param json_file_path: The json file that stores the kfolds.
    :return: The folds in a list of [fold1_train, fold1_test, fold2_train, fold2_test...]
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

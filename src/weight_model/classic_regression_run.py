import logging
import os
from pathlib import Path
from typing import Callable, Union

import numpy as np
import torch

from weight_model.k_fold_builder import load_dataset_folds
from weight_model.mask_mappings import load_dataset
from weight_model.results import ModelRunResults
from weight_model.classic_run_config import ClassicRunConfig
from utils.utils import write_csv_file
from constants.constants import (
    VAL_PREDICTIONS,
    VAL_METRICS,
)
torch.manual_seed(42)


logger = logging.getLogger(__name__)


def scale_using_stored_values(depth_masks: np.ndarray, mean_list: list, std_list: list):
    mean_array = np.array(mean_list).reshape(-1, 1, 1)  # Reshape to match (views, 1, 1)
    std_array = np.array(std_list).reshape(-1, 1, 1)  # Reshape to match (views, 1, 1)

    # Ensure the number of elements in mean and std match the number of dimensions in the depth mask
    if depth_masks.shape[1] != len(mean_list) or depth_masks.shape[1] != len(std_list):
        raise ValueError(
            "Length of mean and std lists must match the number of dimensions in the depth mask."
        )

    # Scale the depth mask
    scaled_mask = (depth_masks - mean_array) / std_array

    return scaled_mask

def transform_input(input_batch: np.ndarray, transforms_function: Callable = None) -> np.ndarray:
    if transforms_function is None:
        return input_batch
    augmented_images = []
    for image in input_batch:
        # Apply augmentations
        image_tensor = torch.from_numpy(image)
        augmented = transforms_function(image_tensor)
        augmented_images.append(augmented.numpy())
    return np.array(augmented_images)


def run_classic_regression_folds(
    data_path: Path, results_dir: Path, run_config: ClassicRunConfig, test_set_path: Path = None
):
    logger.info("Loading folds")
    dataset_folds = load_dataset_folds(data_path)
    if test_set_path:
        X_test, y_test = load_dataset([test_set_path])
    else:
        X_test, y_test = None, None

    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    summary_filename = "summary.csv"
    header = ["MAE_val", "MAPE_val", "RMSE_val", "R2_val", "MAE_test", "MAPE_test", "RMSE_test", "R2_test"] if test_set_path else ["MAE", "MAPE", "RMSE", "R2"]
    metrics = []
    for i, dataset in enumerate(dataset_folds):
        fold_results_dir = results_dir / f"train{i}"
        results = run_classic_regression(
            X_train=dataset.X_train,
            X_val=dataset.X_test,
            y_train=dataset.y_train,
            y_val=dataset.y_test,
            run_config=run_config,
            results_dir=fold_results_dir,
            input_dir=data_path,
            X_test=X_test,
            y_test=y_test
        )
        if type(results) == tuple:
            metrics.append([
                results[0].mae,
                results[0].mape,
                results[0].rmse,
                results[0].r2_score,
                results[1].mae,
                results[1].mape,
                results[1].rmse,
                results[1].r2_score,
            ])
        else:
            metrics.append([results.mae, results.mape, results.rmse, results.r2_score])

    metrics = np.array(metrics)

    # Calculate the mean of the corresponding columns (mean along axis 0)
    metric_average = np.mean(metrics, axis=0)
    metric_std = np.std(metrics, axis=0, ddof=1)
    metrics = metrics.tolist()
    metrics.append(metric_average.tolist())
    metrics.append(metric_std.tolist())

    write_csv_file(results_dir / summary_filename, header=header, rows=metrics)


def run_classic_regression(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    run_config: ClassicRunConfig,
    results_dir: Path,
    input_dir: Path,
    X_test: np.ndarray = None,
    y_test: np.ndarray = None
) -> Union[ModelRunResults, tuple[ModelRunResults, ModelRunResults]]:
    logger.info(f"Running model: {run_config.model_name}")
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    X_train_transformed = transform_input(input_batch=X_train.copy(), transforms_function=run_config.transforms_train)
    X_val_transformed = transform_input(input_batch=X_val.copy(), transforms_function=run_config.transforms_test)

    X_train_flat = X_train_transformed.reshape(len(X_train_transformed), -1)
    X_val_flat = X_val_transformed.reshape(len(X_val_transformed), -1)

    if X_test is not None:
        X_test_transformed = transform_input(input_batch=X_test.copy(), transforms_function=run_config.transforms_test)
        X_test_flat = X_test_transformed.reshape(len(X_test_transformed), -1)
    else:
        X_test_flat = None

    model = run_config.get_model()
    run_config.save_run_args(input_dir=input_dir, results_dir=results_dir)

    # Train the model
    model.fit(X_train_flat, y_train.flatten())

    # Predict on validation set
    predictions = model.predict(X_val_flat)

    val_metrics = ModelRunResults(y_true=y_val.flatten(), y_pred=predictions)
    val_metrics.write_predictions(file_path=results_dir / VAL_PREDICTIONS)
    val_metrics.write_metrics(file_path=results_dir / VAL_METRICS)
    val_metrics.plot_actual_and_predicted_values(
        file_path=results_dir / "actual_vs_predicted_val.png"
    )
    val_metrics.print()

    if X_test_flat is not None:
        predictions = model.predict(X_test_flat)

        test_metrics = ModelRunResults(y_true=y_test.flatten(), y_pred=predictions)
        test_metrics.write_predictions(file_path=results_dir / "test_predictions.csv" )
        test_metrics.write_metrics(file_path=results_dir / "test_metrics.csv")
        test_metrics.plot_actual_and_predicted_values(
            file_path=results_dir / "actual_vs_predicted_test.png"
        )
        test_metrics.print()
        return val_metrics, test_metrics
    else:
        return val_metrics

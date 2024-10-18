import logging
import os
from pathlib import Path

import numpy as np

from weight_model.k_fold_builder import load_dataset_folds
from weight_model.results import ModelRunResults
from weight_model.svr_run_config import SvrRunConfig
from utils.utils import write_csv_file
from constants.constants import (
    VAL_PREDICTIONS,
    VAL_METRICS,
)


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


def run_svr_regression_folds(
    data_path: Path, results_dir: Path, run_config: SvrRunConfig
):
    logger.info("Loading folds")
    dataset_folds = load_dataset_folds(data_path)
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    summary_filename = "summary.csv"
    header = ["MAE", "MAPE", "RMSE", "R2"]
    metrics = []
    for i, dataset in enumerate(dataset_folds):
        fold_results_dir = results_dir / f"train{i}"
        results = run_svr(
            X_train=dataset.X_train,
            y_train=dataset.y_train,
            X_test=dataset.X_test,
            y_test=dataset.y_test,
            run_config=run_config,
            results_dir=fold_results_dir,
            input_dir=data_path,
        )
        metrics.append([results.mae, results.mape, results.rmse, results.r2_score])

    metrics = np.array(metrics)

    # Calculate the mean of the corresponding columns (mean along axis 0)
    metric_average = np.mean(metrics, axis=0)
    metric_std = np.std(metrics, axis=0, ddof=1)
    metrics = metrics.tolist()
    metrics.append(metric_average.tolist())
    metrics.append(metric_std.tolist())

    write_csv_file(results_dir / summary_filename, header=header, rows=metrics)


def run_svr(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    run_config: SvrRunConfig,
    results_dir: Path,
    input_dir: Path,
):
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    X_train_scaled = scale_using_stored_values(
        depth_masks=X_train,
        mean_list=run_config.mean_values,
        std_list=run_config.std_values,
    )
    X_test_scaled = scale_using_stored_values(
        depth_masks=X_test,
        mean_list=run_config.mean_values,
        std_list=run_config.std_values,
    )

    X_train_flat = X_train_scaled.reshape(len(X_train_scaled), -1)
    X_test_flat = X_test_scaled.reshape(len(X_test_scaled), -1)

    model = run_config.get_model()
    run_config.save_run_args(input_dir=input_dir, results_dir=results_dir)

    # Train the model
    model.fit(X_train_flat, y_train.ravel())

    # Predict on validation set
    predictions = model.predict(X_test_flat)

    metrics = ModelRunResults(y_true=y_test.ravel(), y_pred=predictions)
    metrics.write_predictions(file_path=results_dir / VAL_PREDICTIONS)
    metrics.write_metrics(file_path=results_dir / VAL_METRICS)
    metrics.plot_actual_and_predicted_values(
        file_path=results_dir / "actual_vs_predicted.png"
    )

    metrics.print()
    return metrics

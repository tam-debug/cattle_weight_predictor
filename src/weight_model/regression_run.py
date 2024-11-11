import csv
from dataclasses import dataclass
import logging
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from typing import Union

from constants.constants import (
    TRAIN_LOSS_PLOT_FILENAME,
    VAL_LOSS_PLOT_FILENAME,
    VAL_PREDICTIONS,
    VAL_METRICS,
)
from weight_model.k_fold_builder import load_dataset_folds
from weight_model.mask_mappings import load_dataset
from weight_model.results import ModelRunResults, plot_loss
from weight_model.run_config import RunConfig
from utils.utils import write_csv_file

logger = logging.getLogger(__name__)

torch.manual_seed(42)

class CustomDataset(Dataset):
    """
    Custom Dataset for the weight-based predictor.
    """

    def __init__(self, X: torch.Tensor, y: torch.Tensor, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        _y = self.y[idx]

        if self.transform:
            x = self.transform(x)

        return x, _y.clone().detach().to(
            torch.float32
        )  # torch.tensor(_y, dtype=torch.float32)


@dataclass
class TrainingResults:
    training_loss: list[float]
    model: torch.nn.Module

    def __init__(self, model: torch.nn.Module):
        self.training_loss = []
        self.validation_loss = []
        self.model = model


class EarlyStopping:
    def __init__(
        self, patience=10, verbose=False, delta=0.0, path=Path("checkpoint.pth")
    ):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float("inf")

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                logger.info(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            logger.info(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def run_regression_folds(data_path: Path, results_dir: Path, run_config: RunConfig, test_set_path: Path = None):
    """
    Runs regression model on dataset folds.

    :param run_config: Contains parameters for the run including the model.
    :param data_path: Path to the dataset folds JSON file.
    :param results_dir: The directory to store the results.
    """
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

        results = run_regression(
            X_train=dataset.X_train,
            y_train=dataset.y_train,
            X_val=dataset.X_test,
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


def run_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    run_config: RunConfig,
    results_dir: Path,
    input_dir: Path,
    X_test: np.ndarray = None,
    y_test: np.ndarray = None
) -> Union[ModelRunResults, tuple[ModelRunResults, ModelRunResults]]:
    """
    Run the pytorch regression model.

    :param X_train: Features of the train dataset.
    :param y_train: Labels of the train dataset.
    :param X_val: Features of the test dataset.
    :param y_val: Labels of the test dataset.
    :param run_config: The run configuration containing parameters like the model.
    :param results_dir: The directory where the results will be stored.
    :param input_dir: The directory that contained the dataset folds. Used for run args recording.
    :return: The model and the metrics.
    """

    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    model = run_config.get_model()
    optimiser = run_config.get_optimiser(model)
    run_config.save_run_args(
        input_dir=input_dir, results_dir=results_dir, optimiser=optimiser
    )

    train_loader = _prepare_data(
        X=X_train,
        y=y_train,
        transform=run_config.transforms_train,
        shuffle=True,
        stack_three_channels=run_config.stack_three_channels,
        batch_size=run_config.batch_size,
    )
    val_loader = _prepare_data(
        X=X_val,
        y=y_val,
        transform=run_config.transforms_test,
        shuffle=False,
        stack_three_channels=run_config.stack_three_channels,
        batch_size=run_config.batch_size,
    )
    if X_test is not None:
        test_loader = _prepare_data(
            X=X_test,
            y=y_test,
            transform=run_config.transforms_test,
            shuffle=False,
            stack_three_channels=run_config.stack_three_channels,
            batch_size=run_config.batch_size,
        )
    else:
        test_loader = None

    early_stopping = EarlyStopping(
        patience=run_config.patience,
        delta=run_config.delta,
        path=results_dir / "early_stopping.pth",
    )

    training_results = _train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimiser=optimiser,
        loss_function=run_config.loss_function,
        epochs=run_config.epochs,
        early_stopping=early_stopping,
        lr_scheduler=run_config.lr_scheduler,
    )

    _save_loss(
        losses=training_results.training_loss, file_path=results_dir / "train_loss.csv"
    )
    _save_loss(
        losses=training_results.validation_loss,
        file_path=results_dir / "validation_loss.csv",
    )

    plot_loss(
        training_results.training_loss,
        title="Training Loss",
        file_path=results_dir / TRAIN_LOSS_PLOT_FILENAME,
    )
    plot_loss(
        training_results.validation_loss,
        title="Validation Loss",
        file_path=results_dir / VAL_LOSS_PLOT_FILENAME,
    )

    model.load_state_dict(torch.load(early_stopping.path.parent / "early_stopping.pth"))


    predictions = _test(model=model, test_loader=val_loader)

    val_metrics = ModelRunResults(y_true=y_val, y_pred=predictions)
    val_metrics.write_predictions(file_path=results_dir / VAL_PREDICTIONS)
    val_metrics.write_metrics(file_path=results_dir / VAL_METRICS)
    val_metrics.plot_actual_and_predicted_values(
        file_path=results_dir / "actual_vs_predicted_val.png"
    )
    val_metrics.print()

    if test_loader is not None:
        predictions = _test(model=model, test_loader=test_loader)
        test_metrics = ModelRunResults(y_true=y_test, y_pred=predictions)
        test_metrics.write_predictions(file_path=results_dir / "test_predictions.csv")
        test_metrics.write_metrics(file_path=results_dir / "test_metrics.csv")
        test_metrics.plot_actual_and_predicted_values(
            file_path=results_dir / "actual_vs_predicted_test.png"
        )
        test_metrics.print()
        return val_metrics, test_metrics
    else:
        return val_metrics


def _save_loss(losses: list[float], file_path: Path):
    with open(file_path, "w", newline="") as file:
        writer = csv.writer(file)
        for loss in losses:
            writer.writerow([loss])


def _prepare_data(
    X: np.ndarray,
    y: np.ndarray,
    transform: v2.Compose,
    shuffle: bool,
    stack_three_channels: bool,
    batch_size: int,
) -> DataLoader:
    if stack_three_channels:
        X = _stack_three_channels(X)

    if len(y.shape) == 1:
        y = y.reshape(-1, 1)

    dataset = CustomDataset(
        X=torch.from_numpy(X).float(),
        y=torch.from_numpy(y).long(),
        transform=transform,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


def _stack_three_channels(data: np.ndarray) -> np.ndarray:
    """
    Ensure the data has three channels.
    """
    if data.ndim == 3:
        data = np.expand_dims(data, axis=1)
    if data.shape[1] == 1:
        data = np.repeat(data, 3, axis=1)
    elif data.shape[1] == 2:
        zero_channel = np.zeros((data.shape[0], 1, data.shape[2], data.shape[3]))
        data = np.concatenate((data, zero_channel), axis=1)
    elif data.shape[1] in [3, 6, 9]:
        data = data
    else:
        raise ValueError(
            f"Data channels should be 1, 2, 3, 6 or 9. Depth mask shape is {data.shape}"
        )

    return data


def _train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    loss_function,
    early_stopping: EarlyStopping,
    lr_scheduler,  #: Optional[torch.optim.lr_scheduler.LRScheduler],
    epochs: int = 100,
) -> TrainingResults:
    """
    Train the model.
    """
    training_results = TrainingResults(model)
    best_loss = None
    loss_function = loss_function()

    model.train()
    for epoch in range(epochs):
        train_loss = 0.0
        for depth_mask, weights in train_loader:
            if torch.cuda.is_available():
                depth_mask, weights = depth_mask.to("cuda"), weights.to("cuda")
                model.to("cuda")

            # Zero the parameter gradients
            optimiser.zero_grad()

            # Forward pass
            outputs = model(depth_mask)
            loss = loss_function(outputs, weights)

            # Backward pass and optimization
            loss.backward()
            optimiser.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for depth_mask, weights in val_loader:
                if torch.cuda.is_available():
                    depth_mask, weights = depth_mask.to("cuda"), weights.to("cuda")
                    model.to("cuda")

                output = model(depth_mask)
                loss = loss_function(output, weights)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        training_results.training_loss.append(train_loss)
        training_results.validation_loss.append(val_loss)

        logger.info(
            f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            logger.info(f"Early stopping at epoch {epoch} out of {epochs}")
            break

        if lr_scheduler is not None:
            lr_scheduler.step()

    return training_results


def _test(model: torch.nn.Module, test_loader: DataLoader) -> np.ndarray:
    """
    Run the model on test dataset.
    """
    model.eval()
    predictions = np.array([])
    with torch.no_grad():
        for depth_mask, weights in test_loader:
            if torch.cuda.is_available():
                depth_mask, weights = depth_mask.to("cuda"), weights.to("cuda")
                model.to("cuda")

            pred = model(depth_mask)
            predictions = np.append(predictions, pred.cpu().flatten().detach().numpy())

    predictions = predictions.flatten()
    logger.info(predictions)
    return predictions

"""
PyTorch pretrained CNN for weight predictor model.
"""

import os.path
from dataclasses import dataclass
import os
import logging
from pathlib import Path

import torchvision.models
import yaml

import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import v2

from torch.utils.data import DataLoader, Dataset

from constants.constants import (
    TRANSFORM_TRAIN,
    TRANSFORM_TEST,
    TRAIN_LOSS_PLOT_FILENAME,
    VAL_LOSS_PLOT_FILENAME,
    VAL_PREDICTIONS,
    VAL_METRICS,
)
from weight_model.results import ModelRunResults, plot_loss
from weight_model.k_fold_builder import load_dataset_folds
from utils.utils import write_csv_file

logger = logging.getLogger(__name__)


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
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


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


def run_resnet_18_folds(data_path: Path, results_dir: Path):
    """
    Runs Resnet 18 pretrained CNN for dataset folds.

    :param data_path: Path to the dataset folds JSON file.
    :param results_dir: The directory to store the results.
    """
    logger.info("Loading folds")
    dataset_folds = load_dataset_folds(data_path)
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    summary_filename = "summary.csv"
    header = ["MAE", "MAPE", "RMSE", "R2"]
    metrics = []

    for i, dataset in enumerate(dataset_folds):
        fold_results_dir = results_dir / f"train{i}"

        model, results = run_resnet_18(
            X_train=dataset.X_train,
            y_train=dataset.y_train,
            X_test=dataset.X_test,
            y_test=dataset.y_test,
            results_dir=fold_results_dir,
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


def run_pretrained(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model: nn.Module,
    loss_function,
    optimiser: torch.optim.Optimizer,
    epochs: int,
    patience: int,
    delta: float,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    results_dir: Path,
    model_name: str,
) -> tuple[torch.nn.Module, ModelRunResults]:
    """
    Run the pretrained pytorch model.

    :param X_train: Features of the train dataset.
    :param y_train: Labels of the train dataset.
    :param X_test: Features of the test dataset.
    :param y_test: Labels of the test dataset.
    :param model: The model to run.
    :param loss_function: The loss function.
    :param optimiser: The optimiser to use.
    :param epochs: The number of epochs.
    :param results_dir: The directory where the results will be stored.
    :param model_name: The name of the model, saved as one of the run args.
    :return: The model and the metrics.
    """

    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    _save_run_args(
        model_name=model_name,
        optimiser=optimiser,
        loss_function=loss_function,
        epochs=epochs,
        patience=patience,
        delta=delta,
        lr_scheduler=lr_scheduler,
        results_dir=results_dir,
    )

    train_loader = _prepare_data(
        X=X_train, y=y_train, transform=TRANSFORM_TRAIN, shuffle=True
    )
    test_loader = _prepare_data(
        X=X_test, y=y_test, transform=TRANSFORM_TEST, shuffle=False
    )
    early_stopping = EarlyStopping(
        patience=patience, delta=delta, path=results_dir / "checkpoint.pth"
    )

    training_results = _train(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        optimiser=optimiser,
        loss_function=loss_function,
        epochs=epochs,
        early_stopping=early_stopping,
        lr_scheduler=lr_scheduler,
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

    model.load_state_dict(torch.load(early_stopping.path))

    predictions = _test(model=model, test_loader=test_loader)

    metrics = ModelRunResults(y_true=y_test, y_pred=predictions)
    metrics.write_predictions(file_path=results_dir / VAL_PREDICTIONS)
    metrics.write_metrics(file_path=results_dir / VAL_METRICS)
    metrics.plot_actual_and_predicted_values(
        file_path=results_dir / "actual_vs_predicted.png"
    )
    metrics.print()

    return model, metrics


def run_resnet_18(
    X_train, y_train, X_test, y_test, results_dir: Path
) -> tuple[torch.nn.Module, ModelRunResults]:
    """
    Run the ResNet-18 model.

    :param X_train: Features of the train dataset.
    :param y_train: Labels of the train dataset.
    :param X_test: Features of the test dataset.
    :param y_test: Labels of the test dataset.
    :param results_dir: The directory where the results will be stored.
    :return: The model and the metrics.
    """
    model_name = f"resnet18"
    model = _load_resnet()

    loss_function = nn.MSELoss()
    initial_lr = 0.001
    optimiser = torch.optim.Adam(model.parameters(), lr=initial_lr, weight_decay=0)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=10, gamma=0.7)
    epochs = 50
    patience = 10
    delta = 0.01

    return run_pretrained(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model=model,
        loss_function=loss_function,
        optimiser=optimiser,
        epochs=epochs,
        results_dir=results_dir,
        model_name=model_name,
        patience=patience,
        delta=delta,
        lr_scheduler=lr_scheduler,
    )


def _load_resnet() -> nn.Module:
    """
    Load the ResNet model for regression.
    Modified the final fully connected layer to a dense layer.
    """
    model = torchvision.models.resnet18(pretrained=True)

    model.fc = nn.Linear(model.fc.in_features, 128)
    # model.fc = nn.Sequential(
    #     nn.Linear(model.fc.in_features, 128),  # Dense layer with 128 units
    #     nn.ReLU(),  # Activation function
    #     nn.Linear(128, 1),  # Final layer with single output
    # )
    logger.info(model.eval())
    return model


def _load_efficientnet() -> nn.Module:
    """
    Load the ResNet model for regression.
    Modified the final fully connected layer to have a dense layer instead.
    """
    model = torchvision.models.efficientnet_b0(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 1)
    logger.info(model.eval())
    return model


def _prepare_data(
    X: np.ndarray, y: np.ndarray, transform: v2.Compose, shuffle: bool
) -> DataLoader:
    X = _stack_three_channels(X)

    dataset = CustomDataset(
        X=torch.from_numpy(X).float(),
        y=torch.from_numpy(y).long(),
        transform=transform,
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=shuffle)
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
    elif data.shape[1] == 3:
        data = data
    else:
        raise ValueError(
            f"Data channels should be 1, 2 or 3. Depth mask shape is {data.shape}"
        )

    return data


def _train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    loss_function,
    early_stopping: EarlyStopping,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    epochs: int = 100,
) -> TrainingResults:
    """
    Train the model.
    """

    training_results = TrainingResults(model)
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
            loss = loss_function(outputs.squeeze(), weights)

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
                loss = loss_function(output.squeeze(), weights)
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

        # lr_scheduler.step()

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


def _save_run_args(
    model_name: str,
    optimiser: torch.optim.Optimizer,
    loss_function,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    epochs: int,
    results_dir: Path,
    patience: int,
    delta: float,
):
    """
    Save the model run arguments in a YAML file.
    """

    optimiser_info = {
        "optimiser_type": optimiser.__class__.__name__,
        "learning_rate": optimiser.param_groups[0]["lr"],
        "weight_decay": optimiser.param_groups[0].get("weight_decay"),
        "momentum": optimiser.param_groups[0].get("momentum"),
    }

    args = {
        "model_name": model_name,
        "loss_function": loss_function.__class__.__name__,
        "epochs": epochs,
        "optimiser_info": optimiser_info,
        "save_dir": results_dir.as_posix(),
        "patience": patience,
        "delta": delta,
        "lr_scheduler": lr_scheduler.state_dict(),
    }

    with open(results_dir / "run_args.yaml", "w") as file:
        yaml.dump(args, file, default_flow_style=False)

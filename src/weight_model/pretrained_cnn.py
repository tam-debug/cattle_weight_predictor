"""
PyTorch pretrained CNN for weight predictor model.
"""

import os.path
from dataclasses import dataclass
import os
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import v2

from torchvision.models.resnet import ResNet18_Weights
from torch.utils.data import DataLoader, Dataset

from constants.constants import (
    TRANSFORM_TRAIN,
    TRANSFORM_TEST,
    LOSS_PLOT_FILENAME,
    PYTORCH_REPO,
    VAL_PREDICTIONS,
    VAL_METRICS,
)
from weight_model.results import ModelRunResults, plot_training_loss
from k_fold_builder import load_dataset_folds

logger = logging.getLogger(__name__)


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

        return x, torch.tensor(_y, dtype=torch.float32)


@dataclass
class TrainingResults:
    training_loss: list[float]
    model: torch.Module

    def __init__(self, model: torch.Module):
        self.training_loss = []
        self.model = model

def run_resnet_18_folds(data_path: Path, results_dir: Path):
    """
    Runs Resnet 18 pretrained CNN for dataset folds.

    :param data_path: Path to the dataset folds JSON file.
    :param results_dir: The directory to store the results.
    """
    dataset_folds = load_dataset_folds(data_path)

    for i, dataset in enumerate(dataset_folds):
        fold_results_dir = results_dir / f"train{0}"

        run_resnet_18(
            X_train=dataset.X_train,
            y_train=dataset.y_train,
            X_test=dataset.X_test,
            y_test=dataset.y_test,
            results_dir=fold_results_dir
        )



def run_pretrained(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model: nn.Module,
    loss_function,
    optimiser: torch.optim.Optimizer,
    epochs: int,
    results_dir: Path,
) -> tuple[torch.Module, ModelRunResults]:
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
    :return: The model and the metrics.
    """

    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    train_loader = _prepare_data(
        X=X_train, y=y_train, transform=TRANSFORM_TRAIN, shuffle=True
    )
    test_loader = _prepare_data(
        X=X_test, y=y_test, transform=TRANSFORM_TEST, shuffle=False
    )

    training_results = _train(
        model=model,
        train_loader=train_loader,
        optimiser=optimiser,
        loss_function=loss_function,
        epochs=epochs,
    )

    loss_plot_path = results_dir / LOSS_PLOT_FILENAME
    plot_training_loss(training_results.training_loss, file_path=loss_plot_path)

    predictions = _test(model=model, test_loader=test_loader)

    metrics = ModelRunResults(predictions, y_test)
    metrics.write_metrics(file_path=results_dir / VAL_PREDICTIONS)
    metrics.write_metrics(file_path=results_dir / VAL_METRICS)
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
    model = _load_resnet(resnet_number=18)

    loss_function = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 100

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
    )


def _load_resnet(resnet_number: int) -> nn.Module:
    """
    Load the ResNet model for regression.
    Modified the final fully connected layer to output a vector of size 128 (or any desired size)
    """
    model = torch.hub.load(
        PYTORCH_REPO,
        f"resnet{resnet_number}",
        weights=ResNet18_Weights.IMAGENET1K_V1,
    )

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 128),  # Dense layer with 128 units
        nn.ReLU(),  # Activation function
        nn.Linear(128, 1),  # Final layer with single output
    )
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
    if data.ndim == 2:
        data = np.expand_dims(data, axis=1)

    if data.shape[0] == 1:
        data = np.repeat(data, 3, axis=0)
    elif data.shape[0] == 2:
        zero_channel = np.zeros((data.shape[0], data.shape[1]))
        data = np.dstack([data, zero_channel])
    elif data.shape[0] == 3:
        data = data
    else:
        raise ValueError(
            f"Data channels should be 1, 2 or 3. Depth mask shape is {data.shape}"
        )

    return data


def _train(
    model: torch.Module,
    train_loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    loss_function,
    epochs: int = 100,
) -> TrainingResults:
    """
    Train the model.
    """

    training_results = TrainingResults(model)
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
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

            running_loss += loss.item()

        training_results.training_loss.append(running_loss / len(train_loader))

        logger.info(
            f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}"
        )
    return training_results


def _test(model: torch.nn.Module, test_loader: DataLoader) -> np.ndarray:
    """
    Run the model on test dataset.
    """
    model.eval()
    predictions = np.array([])
    for depth_mask, weights in test_loader:
        if torch.cuda.is_available():
            depth_mask, weights = depth_mask.to("cuda"), weights.to("cuda")
            model.to("cuda")

        pred = model(depth_mask)
        predictions = np.append(predictions, pred.cpu().flatten().detach().numpy())

    predictions = predictions.flatten()
    logger.info(predictions)
    return predictions

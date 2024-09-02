from dataclasses import dataclass
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import random_split
from torchvision.transforms import v2
from torchvision.models.resnet import ResNet18_Weights
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class CustomDataset(Dataset):
    def __init__(self, depth_masks, heights, transform=None):
        self.depth_masks = depth_masks
        self.heights = heights
        self.transform = transform

    def __len__(self):
        return len(self.depth_masks)

    def __getitem__(self, idx):
        depth_mask = self.depth_masks[idx]
        height = self.heights[idx]

        depth_mask = torch.tensor(depth_mask, dtype=torch.float32)

        if self.transform:
            depth_mask = self.transform(depth_mask)

        return depth_mask, torch.tensor(height, dtype=torch.float32)


@dataclass
class TrainingResults:
    training_loss: list[float]
    model: torch.Tensor

    def __init__(self, model: torch.Tensor):
        self.training_loss = []
        self.model = model


def split_dataset(
    dataset: Dataset, test_proportion: float = 0.2
) -> tuple[DataLoader, DataLoader]:
    train_size = int((1 - test_proportion) * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader


def run_resnet(depth_masks: np.ndarray, weights: np.ndarray):

    depth_masks = np.expand_dims(depth_masks, axis=1)
    depth_masks = np.repeat(depth_masks, 3, axis=1)

    transform = v2.Compose(
        [v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )
    dataset = CustomDataset(depth_masks, weights, transform=transform)
    train_loader, test_loader = split_dataset(dataset, test_proportion=0.2)

    model = load_resnet()
    loss_function = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

    training_results = train(
        model=model,
        train_loader=train_loader,
        optimiser=optimiser,
        loss_function=loss_function,
        epochs=5,
    )
    plot_training_loss(training_results.training_loss)


def train(
    model: torch.Tensor,
    train_loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    loss_function: torch.nn._Loss,
    epochs: int = 10,
) -> TrainingResults:

    training_results = TrainingResults(model)
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, heights in train_loader:
            if torch.cuda.is_available():
                images, heights = images.to("cuda"), heights.to("cuda")
                model.to("cuda")

            # Zero the parameter gradients
            optimiser.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = loss_function(outputs.squeeze(), heights)

            # Backward pass and optimization
            loss.backward()
            optimiser.step()

            running_loss += loss.item()

        training_results.training_loss.append(running_loss / len(train_loader))

        logger.info(
            f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}"
        )
    return training_results


def load_resnet() -> torch.Tensor:
    model = torch.hub.load(
        "pytorch/vision:v0.10.0", "resnet18", weights=ResNet18_Weights.IMAGENET1K_V1
    )

    # Modify the final fully connected layer to output a vector of size 128 (or any desired size)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 128),  # Dense layer with 128 units
        nn.ReLU(),  # Activation function
        nn.Linear(128, 1),  # Final layer with single output
        nn.Sigmoid(),  # Sigmoid to ensure output is between 0 and 1
    )
    print(model.eval())
    return model


def plot_training_loss(loss: list[float]):
    plt.plot(loss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.show()

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Optional
import yaml

import torch
from torchvision.transforms import v2
import torchvision.models.efficientnet as efficientnet
import torchvision.models.resnet as resnet

from weight_model.custom_model import CNNModel, CNNModel_1

logger = logging.getLogger(__name__)


@dataclass
class RunConfig:
    model_name: str
    model: torch.nn.Module
    loss_function: torch.nn.Module
    initial_lr: float
    optimiser: torch.optim.Optimizer
    epochs: int
    patience: int
    delta: float
    stack_three_channels: bool
    batch_size: int
    lr_scheduler: Optional[torch.optim.lr_scheduler] = None
    transforms_train: Optional[v2.Compose] = None
    transforms_test: Optional[v2.Compose] = None

    def save_run_args(self, results_dir: Path, input_dir: Path):
        """
        Save the model run arguments in a YAML file.

        :param results_dir: The directory where the results are saved.
        :param input_dir: The path to the dataset that was used.
        :return:
        """

        optimiser_info = {
            "optimiser_type": self.optimiser.__class__.__name__,
            "learning_rate": self.optimiser.param_groups[0]["lr"],
            "weight_decay": self.optimiser.param_groups[0].get("weight_decay"),
            "momentum": self.optimiser.param_groups[0].get("momentum"),
        }

        args = {
            "model_name": self.model_name,
            "loss_function": self.loss_function.__class__.__name__,
            "epochs": self.epochs,
            "optimiser_info": optimiser_info,
            "save_dir": results_dir.as_posix(),
            "input_dir": input_dir,
            "patience": self.patience,
            "delta": self.delta,
            "lr_scheduler": (
                self.lr_scheduler.state_dict() if self.lr_scheduler else None
            ),
        }

        with open(results_dir / "run_args.yaml", "w") as file:
            yaml.dump(args, file, default_flow_style=False)


def get_run_config(config_name: str, num_channels: int) -> RunConfig:
    """
    Gets the run configuration for the specified model name.

    :param config_name: The config name associated with the model.
    :param num_channels: The number of channels (required for the custom model).
    :return: The run configuration.
    """
    config_names = ["resnet", "custom", "efficientnet"]
    if config_name not in config_names:
        raise ValueError(f"{config_name} must be either {config_names}")

    if config_name == "resnet":
        model_name = f"resnet_18"
        model = _load_resnet()

        loss_function = torch.nn.MSELoss()
        initial_lr = 0.001
        optimiser = torch.optim.Adam(model.parameters(), lr=initial_lr, weight_decay=0)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimiser, step_size=10, gamma=0.7
        )
        lr_scheduler = None
        epochs = 50
        patience = 10
        delta = 0.01
        batch_size = 64
        stack_three_channels = True
        transforms_train = v2.Compose(
            [
                v2.RandomHorizontalFlip(0.5),
                v2.RandomVerticalFlip(0.5),
                v2.RandomRotation(30),
                v2.RandomResizedCrop(size=(640, 640), scale=(0.8, 1.0)),
                v2.RandomAffine(20),
                v2.RandomPerspective(),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        transforms_test = v2.Compose(
            [v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )

    elif config_name == "efficientnet":
        model_name = f"efficientnet_b0"
        model = _load_resnet()

        loss_function = torch.nn.MSELoss()
        initial_lr = 0.001
        optimiser = torch.optim.Adam(model.parameters(), lr=initial_lr, weight_decay=0)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimiser, step_size=10, gamma=0.7
        )
        lr_scheduler = None
        epochs = 50
        batch_size = 64
        patience = 10
        delta = 0.01
        stack_three_channels = True
        transforms_train = v2.Compose(
            [
                v2.RandomHorizontalFlip(0.5),
                v2.RandomVerticalFlip(0.5),
                v2.RandomRotation(30),
                v2.RandomResizedCrop(size=(640, 640), scale=(0.8, 1.0)),
                v2.RandomAffine(20),
                v2.RandomPerspective(),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )
        transforms_test = v2.Compose(
            [v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )

    elif config_name == "custom":
        model_name = "custom_cnn"
        model = CNNModel(num_channels=num_channels)
        loss_function = torch.nn.MSELoss()
        initial_lr = 0.001
        optimiser = torch.optim.Adam(model.parameters(), lr=initial_lr, weight_decay=0)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimiser, step_size=10, gamma=0.7
        )
        lr_scheduler = None
        epochs = 50
        batch_size = 64
        patience = 20
        delta = 0.01
        stack_three_channels = False
        transforms_train = v2.Compose(
            [
                v2.RandomHorizontalFlip(0.5),
                v2.RandomVerticalFlip(0.5),
                v2.RandomRotation(30),
                v2.RandomResizedCrop(size=(640, 640), scale=(0.8, 1.0)),
                v2.RandomAffine(20),
                v2.RandomPerspective(),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )
        transforms_test = None

    return RunConfig(
        model_name=model_name,
        model=model,
        loss_function=loss_function,
        initial_lr=initial_lr,
        optimiser=optimiser,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        delta=delta,
        lr_scheduler=lr_scheduler,
        stack_three_channels=stack_three_channels,
        transforms_train=transforms_train,
        transforms_test=transforms_test,
    )


def _load_resnet() -> torch.nn.Module:
    """
    Load the ResNet model for regression.
    Modified the final fully connected layer to a dense layer.
    """
    model = resnet.resnet18(weights=resnet.ResNet18_Weights.IMAGENET1K_V1)

    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    # model.fc = nn.Sequential(
    #     nn.Linear(model.fc.in_features, 128),  # Dense layer with 128 units
    #     nn.ReLU(),  # Activation function
    #     nn.Linear(128, 1),  # Final layer with single output
    # )
    logger.info(model.eval())
    return model

def _load_efficientnet() -> torch.nn.Module:
    """
    Load the ResNet model for regression.
    Modified the final fully connected layer to have a dense layer instead.
    """
    model = efficientnet.efficientnet_b0(weights=efficientnet.EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    logger.info(model.eval())
    return model

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Optional
import yaml

import torch
from torchvision.transforms import v2
import torchvision
from typing import Callable

import torchvision.models.resnet as resnet
import torchvision.models.efficientnet as efficientnet

from weight_model.custom_model import CNNModel_4, CNNModel_1, CNNModel_2, CNNModel_3

logger = logging.getLogger(__name__)


@dataclass
class RunConfig:
    model_name: str
    model: Callable
    loss_function: torch.nn.Module
    initial_lr: float
    optimiser: torch.optim.Optimizer
    epochs: int
    patience: int
    delta: float
    stack_three_channels: bool
    batch_size: int
    lr_scheduler: Optional = None
    transforms_train: Optional = None
    transforms_test: Optional = None
    num_channels: int = None


    def save_run_args(self, results_dir: Path, input_dir: Path, optimiser):
        """
        Save the model run arguments in a YAML file.

        :param results_dir: The directory where the results are saved.
        :param input_dir: The path to the dataset that was used.
        :return:
        """

        optimiser_info = {
            "optimiser_type": optimiser.__class__.__name__,
            "learning_rate": optimiser.param_groups[0]["lr"],
            "weight_decay": optimiser.param_groups[0].get("weight_decay"),
            "momentum": optimiser.param_groups[0].get("momentum"),
        }

        args = {
            "model_name": self.model_name,
            "loss_function": self.loss_function.__class__.__name__,
            "epochs": self.epochs,
            "optimiser_info": optimiser_info,
            "save_dir": results_dir.as_posix(),
            "input_dir": input_dir.as_posix(),
            "patience": self.patience,
            "delta": self.delta,
            "lr_scheduler": (
                self.lr_scheduler.state_dict() if self.lr_scheduler else None
            ),
        }

        with open(results_dir / "run_args.yaml", "w") as file:
            yaml.dump(args, file, default_flow_style=False)

    def get_model(self) -> torch.nn.Module:
        if self.num_channels is not None:
            return self.model(self.num_channels)
        else:
            return self.model()

    def get_optimiser(self, model: torch.nn.Module):
        return torch.optim.Adam(model.parameters(), lr=self.initial_lr, weight_decay=0)



def get_run_config(config_name: str, num_channels: int, mean: list[float] = None, std: list[float] = None) -> RunConfig:
    """
    Gets the run configuration for the specified model name.

    :param config_name: The config name associated with the model.
    :param num_channels: The number of channels (required for the custom model).
    :return: The run configuration.
    """
    config_names = ["resnet", "mobilenet", "custom_1", "custom_2"]
    if config_name not in config_names:
        raise ValueError(f"{config_name} must be either {config_names}")

    if config_name == "resnet":
        model_name = f"resnet_18"
        model = _load_resnet

        loss_function = torch.nn.MSELoss()
        initial_lr = 0.001
        optimiser = torch.optim.Adam(_load_resnet().parameters(), lr=initial_lr, weight_decay=0)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimiser, step_size=10, gamma=0.7
        )
        lr_scheduler = None
        epochs = 50
        patience = 15
        delta = 0.01
        batch_size = 64
        stack_three_channels = True
        num_channels = None
        transforms_train = v2.Compose(
            [
                # v2.Resize((640, 640)),
                v2.RandomHorizontalFlip(0.5),
                v2.RandomRotation(30),
                v2.RandomResizedCrop(size=(640, 640), scale=(0.8, 1.0)), # Rescale image
                v2.RandomAffine(degrees=0, translate=(0.2, 0.2)),
                v2.RandomPerspective(),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        transforms_test = v2.Compose(
            [v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )

    elif config_name == "mobilenet":
        model_name = f"mobilenet_v3_small"
        model = _load_mobilenet

        loss_function = torch.nn.MSELoss()
        initial_lr = 0.001
        optimiser = torch.optim.Adam(_load_mobilenet().parameters(), lr=initial_lr, weight_decay=0)
        optimiser = None
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(
        #     optimiser, step_size=10, gamma=0.7
        # )
        lr_scheduler = None
        epochs = 50
        batch_size = 64
        patience = 15
        delta = 0.01
        stack_three_channels = True
        num_channels = None
        transforms_train = v2.Compose(
            [
                # v2.Resize((640, 640)),
                v2.RandomHorizontalFlip(0.5),
                v2.RandomVerticalFlip(0.5),
                v2.RandomRotation(30),
                v2.RandomResizedCrop(size=(640, 640), scale=(0.8, 1.0)),
                v2.RandomAffine(degrees=0, translate=(0.2, 0.2)),
                v2.RandomPerspective(),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )
        transforms_test = v2.Compose(
            [v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )

    elif config_name == "custom_1":
        model_name = "custom_cnn_1"
        model = CNNModel_4
        loss_function = torch.nn.MSELoss()
        initial_lr = 0.001
        optimiser = torch.optim.Adam(CNNModel_4(num_channels).parameters(), lr=initial_lr, weight_decay=0)
        # lr_scheduler = torch.optim.lr_scheduler.steplr(
        #     optimiser, step_size=10, gamma=0.7
        # )
        lr_scheduler = None
        epochs = 50
        batch_size = 64
        patience = 15
        delta = 0.01
        stack_three_channels = False
        num_channels = num_channels
        transforms_train = v2.Compose(
            [
                v2.RandomHorizontalFlip(0.5),
                v2.RandomVerticalFlip(0.5),
                v2.RandomRotation(30),
                v2.RandomResizedCrop(size=(640, 640), scale=(0.8, 1.0)),
                v2.RandomAffine(degrees=0, translate=(0.2, 0.2)),
                v2.RandomPerspective(),
                v2.Normalize(mean=mean, std=std)
            ]
        )
        transforms_test = v2.Compose([v2.Normalize(mean=mean, std=std)])

    elif config_name == "custom_2":
        model_name = "custom_cnn_2"
        model = CNNModel_2
        loss_function = torch.nn.MSELoss()
        initial_lr = 0.001
        optimiser = torch.optim.Adam(CNNModel_2(num_channels).parameters(), lr=initial_lr, weight_decay=0)
        # lr_scheduler = torch.optim.lr_scheduler.steplr(
        #     optimiser, step_size=10, gamma=0.7
        # )
        lr_scheduler = None
        epochs = 50
        batch_size = 64
        patience = 15
        delta = 0.01
        stack_three_channels = False
        num_channels = num_channels
        transforms_train = v2.Compose(
            [
                v2.RandomHorizontalFlip(0.5),
                v2.RandomVerticalFlip(0.5),
                v2.RandomRotation(30),
                v2.RandomResizedCrop(size=(640, 640), scale=(0.8, 1.0)),
                v2.RandomAffine(degrees=0, translate=(0.2, 0.2)),
                v2.RandomPerspective(),
                v2.Normalize(mean=mean, std=std)
            ]
        )
        transforms_test = v2.Compose([v2.Normalize(mean=mean, std=std)])

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
        num_channels=num_channels
    )


def _load_resnet() -> torch.nn.Module:
    """
    Load the ResNet model for regression.
    Modified the final fully connected layer to a dense layer.
    """
    model = resnet.resnet18(weights=resnet.ResNet18_Weights.IMAGENET1K_V1)

    model.fc = torch.nn.Linear(model.fc.in_features, 1)

    logger.info(model.eval())
    return model

def _load_mobilenet() -> torch.nn.Module:
    """
    Load the Mobilenet model for regression.
    Modified the final fully connected layer to have a dense layer instead.
    """
    model = torchvision.models.mobilenet_v3_small(weights=torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    model.classifier = torch.nn.Linear(model.classifier[0].in_features, 1)
    logger.info(model.eval())
    return model

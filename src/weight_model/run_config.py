from abc import abstractmethod, ABC
from dataclasses import dataclass
import logging
from pathlib import Path
import yaml

import torch
from torchvision.transforms import v2
import torchvision
from typing import Callable

import torchvision.models.resnet as resnet

from weight_model.custom_model import CNNModel_4, CNNModel_2

logger = logging.getLogger(__name__)

DATA_AUGMENTATIONS = [
    v2.RandomHorizontalFlip(0.5),
    v2.RandomRotation(30),
    v2.RandomResizedCrop(size=(640, 640), scale=(0.8, 1.0)),
    v2.RandomAffine(degrees=0, translate=(0.2, 0.2)),
    v2.RandomPerspective()
]

class PretrainedModel(ABC):
    """
    Interface for a PretrainModel object.
    """
    model: torch.nn.Module

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_conv1(self) -> torch.nn.Module:
        pass

    @abstractmethod
    def get_final_layer_in_features(self) -> int:
        pass

    @abstractmethod
    def set_first_layer(self, new_layer: torch.nn.Module):
        pass

    @abstractmethod
    def set_final_layer(self, new_layer: torch.nn.Module):
        pass


@dataclass
class RunConfig:
    model_name: str
    exclude_attr_from_run_args: list[str]
    loss_function: torch.nn.Module
    initial_lr: float
    optimiser: torch.optim.Optimizer
    epochs: int
    patience: int
    delta: float
    stack_three_channels: bool
    batch_size: int
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler
    transforms_train: v2.Compose
    transforms_test: v2.Compose
    num_channels: int

    @abstractmethod
    def get_model(self):
        pass

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
        args = vars(self).copy()
        args["save_dir"] = results_dir.as_posix()
        args["input_dir"] = input_dir.as_posix()
        args["optimiser_info"] = optimiser_info

        for key in self.exclude_attr_from_run_args:
            if key in args.keys():
                del args[key]

        with open(results_dir / "run_args.yaml", "w") as file:
            yaml.dump(args, file, default_flow_style=False)

    def get_kwargs(self):
        exclude_keys = [
            "model_name",
            "exclude_attr_from_run_args",
        ]
        kwargs = vars(self).copy()
        for key in exclude_keys:
            if key in kwargs.keys():
                del kwargs[key]
        return kwargs

    def get_optimiser(self, model: torch.nn.Module):
        return torch.optim.Adam(model.parameters(), lr=self.initial_lr, weight_decay=0)


@dataclass
class CustomCNNRunConfig(RunConfig):
    model: torch.nn.Module

    def get_model(self) -> torch.nn.Module:
        if self.num_channels is not None:
            return self.model(self.num_channels)
        else:
            return self.model()


@dataclass
class PretrainedCNNRunConfig(RunConfig):
    model: Callable[[], PretrainedModel]

    def get_model(self) -> torch.nn.Module:
        pretrained_model = self.model()
        conv1 = pretrained_model.get_conv1()
        final_input_features_num = pretrained_model.get_final_layer_in_features()

        if self.num_channels > 3:
            if self.num_channels % 3 != 0:
                raise ValueError(
                    "The number of input channels must be a multiple of 3."
                )

            pretrained_conv1_weights = conv1.weight

            # Get properties from the original conv1 layer
            out_channels = conv1.out_channels
            kernel_size = conv1.kernel_size
            stride = conv1.stride
            padding = conv1.padding
            bias = conv1.bias is not None

            # Create a new conv1 layer with dynamically retrieved parameters and num_input_channels
            new_conv1 = torch.nn.Conv2d(
                in_channels=self.num_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            )

            # Copy the weights of the original 3 channels to all input channels dynamically
            with torch.no_grad():
                for i in range(0, self.num_channels, 3):
                    new_conv1.weight[:, i : i + 3, :, :] = pretrained_conv1_weights

            # Replace the old conv1 layer with the new one
            pretrained_model.set_first_layer(new_conv1)

        new_final_layer = torch.nn.Linear(final_input_features_num, 1)
        pretrained_model.set_final_layer(new_final_layer)

        model = pretrained_model.model

        logger.info(model.eval())
        return model

def get_normalise(num_channels: int = None, mean: list = None, std: list = None):
    mean = [0.485, 0.456, 0.406] if not mean else mean
    std = [0.229, 0.224, 0.225] if not std else std
    if num_channels is None or num_channels <= 3:
        mean, std = mean, std
    elif num_channels == 6:
        mean = mean*2
        std = std*2
    elif num_channels == 9:
        mean = mean*3
        std = std*3
    else:
        raise ValueError(f"Number of channels must be 1, 2, 3, 6 or 9 and not {num_channels}")
    return v2.Normalize(mean=mean, std=std)


def get_run_config(
    config_name: str,
    num_channels: int,
    mean: list[float] = None,
    std: list[float] = None,
) -> RunConfig:
    """
    Gets the run configuration for the specified model name.

    :param config_name: The config name associated with the model.
    :param num_channels: The number of channels (required for the custom model).
    :return: The run configuration.
    """
    config_names = [
        "resnet",
        "mobilenet",
        "custom_4",
        "custom_2",
    ]
    if config_name not in config_names:
        raise ValueError(f"{config_name} must be either {config_names}")

    default_args = {
        "loss_function": torch.nn.MSELoss,
        "initial_lr": 0.001,
        "optimiser": torch.optim.Adam,
        "epochs": 50,
        "patience": 15,
        "delta": 0.01,
        "batch_size": 64,
        "lr_scheduler": None,
        "exclude_attr_from_run_args": ["mean_values", "std_values", "transforms_train", "transforms_test", "exclude_attr_from_run_args"]
    }
    num_channels = 3 if num_channels is None else num_channels


    if config_name == "resnet":
        transforms_test = [v2.ToImage(), get_normalise(num_channels)]
        transforms_train = DATA_AUGMENTATIONS.copy()
        transforms_train.extend(transforms_test)

        config_args = {
            **default_args,
            "model_name": "resnet_18",
            "model": ResNet18,
            "stack_three_channels": True,
            "patience": 15,
            "num_channels": num_channels,
            "transforms_train": v2.Compose(transforms_train),
            "transforms_test": v2.Compose(transforms_test),
        }

        return PretrainedCNNRunConfig(**config_args)

    elif config_name == "mobilenet":
        transforms_test = [v2.ToImage(), get_normalise(num_channels)]
        transforms_train = DATA_AUGMENTATIONS.copy()
        transforms_train.extend(transforms_test)

        config_args = {
            **default_args,
            "model_name": "mobilenet_v3_small",
            "patience": 20,
            "model": MobileNetV3Small,
            "stack_three_channels": True,
            "num_channels": num_channels,
            "transforms_train": v2.Compose(transforms_train),
            "transforms_test": v2.Compose(transforms_test),
        }

        return PretrainedCNNRunConfig(**config_args)

    elif config_name == "custom_4":
        transforms_test = [v2.Normalize(mean=mean, std=std)]
        transforms_train = DATA_AUGMENTATIONS.copy()
        transforms_train.extend(transforms_test)

        config_args = {
            **default_args,
            "model_name": "custom_cnn_4",
            "model": CNNModel_4,
            "stack_three_channels": False,
            "num_channels": num_channels,
            "transforms_train": v2.Compose(transforms_train),
            "transforms_test": v2.Compose(transforms_test),
        }
        return CustomCNNRunConfig(**config_args)

    elif config_name == "custom_2":
        transforms_test = [v2.Normalize(mean=mean, std=std)]
        transforms_train = DATA_AUGMENTATIONS.copy()
        transforms_train.extend(transforms_test)

        config_args = {
            **default_args,
            "model_name": "custom_cnn_2",
            "model": CNNModel_2,
            "stack_three_channels": False,
            "num_channels": num_channels,
            "transforms_train": v2.Compose(transforms_train),
            "transforms_test": v2.Compose(transforms_test),
        }
        return CustomCNNRunConfig(**config_args)


class ResNet18(PretrainedModel):
    def __init__(self):
        self.model = resnet.resnet18(weights=resnet.ResNet18_Weights.IMAGENET1K_V1)

    def get_conv1(self) -> torch.nn.Module:
        return self.model.conv1

    def get_final_layer_in_features(self) -> int:
        return self.model.fc.in_features

    def set_first_layer(self, new_layer: torch.nn.Module):
        self.model.conv1 = new_layer

    def set_final_layer(self, new_layer: torch.nn.Module):
        self.model.fc = new_layer


class MobileNetV3Small(PretrainedModel):
    def __init__(self):
        self.model = torchvision.models.mobilenet_v3_small(
            weights=torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )

    def get_conv1(self) -> torch.nn.Module:
        return self.model.features[0][0]

    def get_final_layer_in_features(self) -> int:
        return self.model.classifier[0].in_features

    def set_first_layer(self, new_layer: torch.nn.Module):
        self.model.features[0][0] = new_layer

    def set_final_layer(self, new_layer: torch.nn.Module):
        self.model.classifier = new_layer

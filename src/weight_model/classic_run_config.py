from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Union
import yaml

from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import (
    BaggingRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
)
import albumentations as A

# Define the augmentations
AUGMENTATIONS = [
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=30, p=1.0),
    A.RandomResizedCrop(height=640, width=640, scale=(0.8, 1.0), p=1.0),
    A.Affine(translate_percent=(0.2, 0.2), p=1.0),
    A.Perspective(p=1.0),
]


@dataclass
class ClassicRunConfig:
    model_name: str
    exclude_attr_from_run_args: list[str]
    transforms_train: A.Compose
    transforms_test: A.Compose

    @abstractmethod
    def get_model(self):
        pass

    def save_run_args(self, results_dir: Path, input_dir: Path):
        args = vars(self).copy()
        args["save_dir"] = results_dir.as_posix()
        args["input_dir"] = input_dir.as_posix()

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


@dataclass
class SvrRunConfig(ClassicRunConfig):
    kernel: str
    degree: int
    gamma: str
    coef0: float
    tol: float
    C: float
    epsilon: float
    shrinking: bool
    cache_size: int
    max_iter: int

    def get_model(self):
        model_args = self.get_kwargs()
        return SVR(**model_args)


@dataclass
class LinearRunConfig(ClassicRunConfig):
    fit_intercept: bool = True
    copy_X: bool = True
    n_jobs: Union[int, None] = None
    positive: bool = False

    def get_model(self):
        model_args = self.get_kwargs()
        return LinearRegression(**model_args)


@dataclass
class RidgeRunConfig(ClassicRunConfig):
    fit_intercept: bool = True
    copy_X: bool = True
    max_iter: int = None
    tol: float = 0.0001
    solver: str = "auto"
    positive: bool = False
    random_state: int = None

    def get_model(self):
        model_args = self.get_kwargs()
        return Ridge(**model_args)


@dataclass
class BaggingRunConfig(ClassicRunConfig):
    estimator_run_config: ClassicRunConfig = None
    n_estimators: int = 50
    max_samples: int = 1
    max_features: int = 1
    bootstrap: bool = True
    bootstrap_features: bool = False
    oob_score: bool = False
    warm_start: bool = False
    n_jobs: int = None
    random_state: int = None

    def get_model(self):
        model_args = self.get_kwargs()
        model_args["estimator"] = (
            self.estimator_run_config.get_model() if self.estimator_run_config else None
        )
        return BaggingRegressor(**model_args)


@dataclass
class RandomForestRunConfig(ClassicRunConfig):
    n_estimators: int = 50
    criterion: str = "squared_error"
    max_depth: int = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    min_weight_fraction_leaf: float = 0.0
    max_features: float = 1
    max_leaf_nodes: int = None
    min_impurity_decrease: float = 0.0
    bootstrap: bool = True
    oob_score: bool = False
    n_jobs: int = None
    random_state: int = None
    warm_start: bool = False
    ccp_alpha: float = 0.0
    max_samples: int = None

    def get_model(self):
        model_args = self.get_kwargs()
        return RandomForestRegressor(**model_args)


@dataclass
class GradientBoostingRunConfig(ClassicRunConfig):
    loss: str = "squared_error"
    learning_rate: float = 0.1
    n_estimators: int = 50
    subsample: float = 1
    criterion: str = "friedman_mse"
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    min_weight_fraction_leaf: float = 0.0
    max_depth: int = 3
    min_impurity_decrease: float = 0.0
    random_state: int = None
    max_features: int = None
    alpha: float = 0.9
    max_leaf_nodes: int = None
    warm_start: bool = False
    validation_fraction: float = 0.1
    n_iter_no_change: int = None
    tol: float = 0.0001
    ccp_alpha: float = 0.0

    def get_model(self):
        model_args = self.get_kwargs()
        return GradientBoostingRegressor(**model_args)


def get_classic_run_config(
    mean: list[float], std: list[float], config_name: str = "svr"
) -> ClassicRunConfig:
    config_names = [
        "svr",
        "linear",
        "ridge",
        "bagging",
        "random_forest",
        "gradient_boosting",
    ]
    exclude_from_run_args = ["mean_values", "std_values"]

    if config_name not in config_names:
        raise ValueError(f"{config_name} must be either {config_names}")

    transforms_test = [A.Normalize(mean=mean, std=std)]
    transforms_train = AUGMENTATIONS
    transforms_train.extend(transforms_test)

    if config_name == "svr":
        model_name = "SVR"
        kernel = "poly"
        degree = 3
        gamma = "scale"
        coef0 = 0.0
        tol = 0.001
        C = 1.0
        epsilon = 0.1
        shrinking = True
        cache_size = 200
        max_iter = -1

        return SvrRunConfig(
            model_name=model_name,
            transforms_train=A.Compose(transforms_train),
            transforms_test=A.Compose(transforms_test),
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            C=C,
            epsilon=epsilon,
            shrinking=shrinking,
            cache_size=cache_size,
            max_iter=max_iter,
            exclude_attr_from_run_args=exclude_from_run_args,
        )
    elif config_name == "linear":
        model_name = "LinearRegression"
        fit_intercept = True
        copy_X = True
        n_jobs = None
        positive = False

        return LinearRunConfig(
            model_name=model_name,
            transforms_train=A.Compose(transforms_train),
            transforms_test=A.Compose(transforms_test),
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            n_jobs=n_jobs,
            positive=positive,
            exclude_attr_from_run_args=exclude_from_run_args,
        )

    elif config_name == "ridge":
        model_name = "Ridge"
        fit_intercept = True
        copy_X = True
        positive = False
        max_iter = None
        tol = 0.0001
        solver = "auto"
        random_state = None

        return RidgeRunConfig(
            model_name=model_name,
            transforms_train=A.Compose(transforms_train),
            transforms_test=A.Compose(transforms_test),
            exclude_attr_from_run_args=exclude_from_run_args,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            positive=positive,
            max_iter=max_iter,
            tol=tol,
            solver=solver,
            random_state=random_state,
        )

    elif config_name == "bagging":
        model_name = "BaggingRegressor"
        # By default, uses the Decision Tree Regressor as base estimator
        return BaggingRunConfig(
            model_name=model_name,
            transforms_train=A.Compose(transforms_train),
            transforms_test=A.Compose(transforms_test),
            exclude_attr_from_run_args=exclude_from_run_args,
        )
    elif config_name == "random_forest":
        model_name = "RandomForestRegressor"
        n_estimators = 50
        return RandomForestRunConfig(
            model_name=model_name,
            transforms_train=A.Compose(transforms_train),
            transforms_test=A.Compose(transforms_test),
            exclude_attr_from_run_args=exclude_from_run_args,
            n_estimators=n_estimators,
        )
    elif config_name == "gradient_boosting":
        model_name = "GradientBoostingRegressor"
        return GradientBoostingRunConfig(
            model_name=model_name,
            transforms_train=A.Compose(transforms_train),
            transforms_test=A.Compose(transforms_test),
            exclude_attr_from_run_args=exclude_from_run_args,
        )

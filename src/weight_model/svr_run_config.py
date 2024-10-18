from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Union
import yaml

from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression


@dataclass
class ClassicRunConfig:
    model_name: str
    mean_values: list[float]
    std_values: list[float]
    exclude_attr_from_run_args: list[str]

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


@dataclass
class SvrRunConfig(ClassicRunConfig):
    # model_name: str
    # mean_values: list[float]
    # std_values: list[float]
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
        return SVR(
            kernel=self.kernel,
            degree=self.degree,
            gamma=self.gamma,
            coef0=self.coef0,
            tol=self.tol,
            C=self.C,
            epsilon=self.epsilon,
            shrinking=self.shrinking,
            cache_size=self.cache_size,
            max_iter=self.max_iter,
        )


@dataclass
class LinearRunConfig(ClassicRunConfig):
    fit_intercept: bool = True
    copy_X: bool = True
    n_jobs: Union[int, None] = None
    positive: bool = False

    def get_model(self):
        return LinearRegression(
            fit_intercept=self.fit_intercept,
            copy_X=self.copy_X,
            n_jobs=self.n_jobs,
            positive=self.positive,
        )


def get_classical_run_config(
    mean: list[float], std: list[float], config_name: str = "svr"
) -> ClassicRunConfig:
    config_names = ["svr", "linear"]
    exclude_from_run_args = ["mean", "std"]

    if config_name not in config_names:
        raise ValueError(f"{config_name} must be either {config_names}")

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
            mean_values=mean,
            std_values=std,
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
            mean_values=mean,
            std_values=std,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            n_jobs=n_jobs,
            positive=positive,
            exclude_attr_from_run_args=exclude_from_run_args,
        )

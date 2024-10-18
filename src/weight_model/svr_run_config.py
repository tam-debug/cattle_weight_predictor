from dataclasses import dataclass
from pathlib import Path
import yaml

from sklearn.svm import SVR


@dataclass
class SvrRunConfig:
    model_name: str
    mean_values: list[float]
    std_values: list[float]
    kernel: str = None
    degree: int = 3
    gamma: str = "scale"
    coef0: float = 0.0
    tol: float = 0.001
    C: float = 1.0
    epsilon: float = 0.1
    shrinking: bool = True
    cache_size: int = 200
    max_iter: int = -1

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

    def save_run_args(self, results_dir: Path, input_dir: Path):
        args = {
            "model_name": self.model_name,
            "save_dir": results_dir.as_posix(),
            "input_dir": input_dir.as_posix(),
            "kernel": self.kernel,
            "degree": self.degree,
            "gamma": self.gamma,
            "coef0": self.coef0,
            "tol": self.tol,
            "C": self.C,
            "epsilon": self.epsilon,
            "shrinking": self.shrinking,
            "cache_size": self.cache_size,
            "max_iter": self.max_iter,
        }

        with open(results_dir / "run_args.yaml", "w") as file:
            yaml.dump(args, file, default_flow_style=False)


def get_svr_run_config(
    mean: list[float], std: list[float], config_name: str = "svr"
) -> SvrRunConfig:
    config_names = ["svr"]

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
        )

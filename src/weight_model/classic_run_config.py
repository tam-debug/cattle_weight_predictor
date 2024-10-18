from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Union
import yaml

from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor


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
        return Ridge(
            fit_intercept=self.fit_intercept,
            copy_X=self.copy_X,
            positive=self.positive,
            max_iter=self.max_iter,
            tol=self.tol,
            solver=self.solver
        )

@dataclass
class BaggingRunConfig(ClassicRunConfig):
    estimator_run_config: ClassicRunConfig = None
    n_estimators: int = 10
    max_samples: int = 1
    max_features: int = 1
    bootstrap: bool = True
    bootstrap_features: bool = False
    oob_score: bool = False
    warm_start: bool = False
    n_jobs: int = None
    random_state: int = None

    def get_model(self):
        estimator = self.estimator_run_config.get_model() if self.estimator_run_config else None
        return BaggingRegressor(
            estimator=self.estimator,
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            bootstrap_features=self.bootstrap_features,
            oob_score=self.oob_score,
            warm_start=self.warm_start,
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )

@dataclass
class RandomForestRunConfig(ClassicRunConfig):
    n_estimators: int = 100
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
        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            bootstrap=self.bootstrap,
            oob_score=self.oob_score,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            warm_start=self.warm_start,
            ccp_alpha=self.ccp_alpha,
            max_samples=self.max_samples
        )

def get_classic_run_config(
    mean: list[float], std: list[float], config_name: str = "svr"
) -> ClassicRunConfig:
    config_names = ["svr", "linear", "ridge", "bagging", "random_forest"]
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
            mean_values=mean,
            std_values=std,
            exclude_attr_from_run_args=exclude_from_run_args,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            positive=positive,
            max_iter=max_iter,
            tol=tol,
            solver=solver,
            random_state=random_state
        )

    elif config_name == "bagging":
        model_name = "BaggingRegressor"
        # By default uses the Decision Tree Regressor as base estimator
        return BaggingRunConfig(
            model_name=model_name,
            mean_values=mean,
            std_values=std,
            exclude_attr_from_run_args=exclude_from_run_args,
        )
    elif config_name == "random_forest":
        model_name = "RandomForestRegressor"
        return RandomForestRunConfig(
            model_name=model_name,
            mean_values=mean,
            std_values=std,
            exclude_attr_from_run_args=exclude_from_run_args,
        )



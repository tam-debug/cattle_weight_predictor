"""
Methods for storing and visualising results for the weight model.
"""

from pathlib import Path
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
    mean_absolute_percentage_error,
)

from utils.utils import write_csv_file


class ModelRunResults:
    """
    Results from the weight model run.
    Contains the metrics, model predictions and actual labels.
    """

    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray):
        self.mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
        self.mape = mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred)
        self.rmse = root_mean_squared_error(y_true=y_true, y_pred=y_pred)
        self.r2_score = r2_score(y_true=y_true, y_pred=y_pred)
        self.predictions = y_pred
        self.actual = y_true

    def print(self):
        """
        Print the metrics from the model run.
        """
        attributes = vars(self)
        exclude_attributes = ["predictions", "actual"]

        filtered_attributes = {
            attr: value
            for attr, value in attributes.items()
            if attr not in exclude_attributes
        }
        pprint(filtered_attributes)

    def write_metrics(self, file_path: Path):
        """
        Writes the metrics to a CSV file.
        :param file_path: The CSV file path to write to.
        """
        header = ["MAE", "MAPE", "RMSE", "R2"]
        rows = [[self.mae, self.mape, self.rmse, self.r2_score]]
        write_csv_file(file_path=file_path, header=header, rows=rows)

    def write_predictions(self, file_path: Path):
        """
        Writes the predictions to a CSV file.
        :param file_path: The CSV file path to write to.
        """
        header = ["Actual", "Predicted"]
        rows = np.column_stack((self.actual, self.predictions))
        write_csv_file(file_path=file_path, header=header, rows=rows)

    def plot_actual_and_predicted_values(self, file_path: Path):
        plt.figure(figsize=(8, 6))

        # Plot actual vs predicted values
        plt.scatter(self.actual, self.predictions, color='blue', label='Predicted vs Actual')

        # Plot a line of equality
        plt.plot([min(self.actual), max(self.actual)], [min(self.actual), max(self.actual)], color='red', linestyle='--',
                 label='Perfect Prediction')

        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Values')
        plt.legend()

        plt.savefig(file_path)
        plt.close()



def plot_loss(loss: list[float], title: str, file_path: Path = None):
    """
    Plots the loss where it can be optionally saved.

    :param loss: The loss values to plot.
    :param file_path: The file path to save the plot to.
    """
    plt.plot(loss)
    plt.yscale("log")
    plt.xlabel("Epoch starting at 5")
    plt.ylabel("Loss")
    plt.title(title)
    if file_path:
        plt.savefig(file_path)
    else:
        plt.show()
    plt.close()

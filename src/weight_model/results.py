from pathlib import Path
from pprint import pprint
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error, mean_absolute_percentage_error

from utils.utils import write_csv_file

class Metrics:
    def __init__(self, y_true, y_pred):
        self.mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
        self.mape = mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred)
        self.rmse = root_mean_squared_error(y_true=y_true, y_pred=y_pred)
        self.r2_score = r2_score(y_true=y_true, y_pred=y_pred)

    def print(self):
        pprint(vars(self))

def plot_training_loss(loss: list[float], file_path: Path = None):
    plt.plot(loss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    if file_path:
        plt.savefig(file_path)
    else:
        plt.show()
def write_metrics(file_path: Path, metrics: Metrics):
    header = ["MAE", "MAPE", "RMSE", "R2"]
    rows = [[metrics.mae, metrics.mape, metrics.rmse, metrics.r2_score]]
    write_csv_file(file_path=file_path, header=header, rows=rows)

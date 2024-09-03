from pprint import pprint
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error, mean_absolute_percentage_error

class Metrics:
    def __init__(self, y_true, y_pred):
        self.mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
        self.mape = mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred)
        self.rmse = root_mean_squared_error(y_true=y_true, y_pred=y_pred)
        self.r2_score = r2_score(y_true=y_true, y_pred=y_pred)

    def print(self):
        pprint(vars(self))



import numpy as np
import pandas as pd

from pyutils.modeling.config import LOG_TRANSFORM


class Transformers:
    @staticmethod
    def log_transform(data: pd.DataFrame, label):
        if LOG_TRANSFORM:
            data[label] = np.log(data[label])

    @staticmethod
    def inverse_log_transform(data, log_transform=LOG_TRANSFORM):
        if log_transform:
            data = np.exp(data)
        return data

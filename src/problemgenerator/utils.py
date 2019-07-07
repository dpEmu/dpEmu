import json
import numpy as np


def load_ocr_error_params(path_to_error_params):
    return json.load(open(path_to_error_params))


def normalize_ocr_error_params(params):
    return {k: (v[0], normalize_probs(v[1])) for k, v in params.items()}


def normalize_probs(probs):
    total = sum(probs)
    return [prob / total for prob in probs]


def to_time_series_x_y(data, x_length):
    """
    Convert time series data to pairs of x, y where x is a vector of x_length
    consecutive observations and y is the observation immediately following x.
    """
    x = np.array([data[i - x_length:i] for i in range(x_length, data.shape[0])])
    y = np.array([data[i] for i in range(x_length, data.shape[0])])
    return x, y


def first_dimension_length(array):
    if type(array) is list:
        return len(array)
    else:
        return array.shape[0]

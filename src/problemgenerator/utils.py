import json

from random import randint

import numpy as np


def load_ocr_error_params(path_to_error_params):
    """Loads error parameters from a JSON-file.

    Args:
        path_to_error_params (str): A string containing the relative or absolute path to the file.

    Returns:
        dict: A Python dictionary.
    """
    return json.load(open(path_to_error_params))


def normalize_ocr_error_params(params):
    """Normalises a numerical weights associated with a character's OCR-error likelihoods.

    For every character found in the dict, the value associated with it
    is a list containing numerical weights. These weights are normalised
    so that they sum to 1, and can thus be used as probabilities.

    Args:
        params (dict): A dict containing character-list pairs.

    Returns:
        dict: A dict containing normalised probabilities for every character.
    """
    return {k: (v[0], normalize_probs(v[1])) for k, v in params.items()}


def normalize_probs(weights):
    """Normalises a list of numerical values (weights) into probabilities.

    Every weight in the list is assigned a probability proportional to its
    value divided by the sum of all values.

    Args:
        weights (list): A list of numerical values

    Returns:
        list: A list containing values which sum to 1.
    """
    total = sum(weights)
    return [weight / total for weight in weights]


def to_time_series_x_y(data, x_length):
    """
    Convert time series data to pairs of x, y where x is a vector of x_length
    consecutive observations and y is the observation immediately following x.

    Args:
        data ([type]):
        x_length (int): Length of the x vector.

    Returns:
        [type]: [description]
    """
    x = np.array([data[i - x_length:i] for i in range(x_length, data.shape[0])])
    y = np.array([data[i] for i in range(x_length, data.shape[0])])
    return x, y


def first_dimension_length(array):
    """Returns the length of the first dimension of the provided array or list.

    Args:
        array (list or numpy.ndarray): An array.

    Returns:
        int: The length of the first dimension of the array.
    """
    if type(array) is list:
        return len(array)
    else:
        return array.shape[0]


def generate_random_dict_key(dct, prefix):
    """Generates a random string that is not already in the dict.

    Args:
        dct (dict): A Python dictionary.
        prefix (str): A prefix for the random key.

    Returns:
        str: A randomly generated key.
    """

    key = prefix
    while key in dct:
        key += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[randint(0, 25)]
    return key

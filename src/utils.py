from datetime import datetime
from os.path import dirname, isfile, join, realpath

import numpy as np
from sklearn.datasets import load_digits, fetch_openml
from sklearn.model_selection import train_test_split


def load_digits_as_npy():
    digits = load_digits()
    path_to_data = join(get_project_root(), "{}/{}.{}".format("data", "digits_data", "npy"))
    path_to_labels = join(get_project_root(), "{}/{}.{}".format("data", "digits_labels", "npy"))
    _save_data_and_labels(digits["data"], digits["target"], path_to_data, path_to_labels)
    return path_to_data, path_to_labels


def load_mnist_as_npy(train_size):
    mnist = fetch_openml("mnist_784")
    if train_size == mnist["data"].shape[0]:
        data = mnist["data"]
        labels = mnist["target"].astype(int)
    else:
        data, _, labels, _ = train_test_split(mnist["data"], mnist["target"].astype(int), train_size=train_size,
                                              random_state=42)
    path_to_data = join(get_project_root(), "{}/{}_{}.{}".format("data", "mnist_data", train_size, "npy"))
    path_to_labels = join(get_project_root(), "{}/{}_{}.{}".format("data", "mnist_labels", train_size, "npy"))
    _save_data_and_labels(data, labels, path_to_data, path_to_labels)
    return path_to_data, path_to_labels


def _save_data_and_labels(data, labels, path_to_data, path_to_labels):
    if not isfile(path_to_data):
        np.save(path_to_data, data)
    if not isfile(path_to_labels):
        np.save(path_to_labels, labels)


def generate_unique_path(folder_name, extension, prefix=None):
    root_folder = get_project_root()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    if prefix:
        return join(root_folder, "{}/{}_{}.{}".format(folder_name, prefix, timestamp, extension))
    return join(root_folder, "{}/{}.{}".format(folder_name, timestamp, extension))


def get_project_root():
    return dirname(dirname(realpath(__file__)))


def expand_parameter_to_linspace(param):
    if len(param) == 1:
        param = (param[0], param[0], 1)
    return np.linspace(*param)

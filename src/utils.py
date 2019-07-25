import pickle
from datetime import datetime
from os.path import dirname, isfile, join, realpath

import numpy as np
from sklearn.datasets import load_digits, fetch_openml, fetch_20newsgroups
from sklearn.model_selection import train_test_split


def load_digits_as_npy():
    """[summary]

    [extended_summary]

    Returns:
        [type]: [description]
    """
    digits = load_digits()
    path_to_data = join(get_project_root(), "{}/{}.{}".format("data", "digits_data", "npy"))
    path_to_labels = join(get_project_root(), "{}/{}.{}".format("data", "digits_labels", "npy"))
    _save_data_and_labels(digits["data"], digits["target"], path_to_data, path_to_labels)
    return path_to_data, path_to_labels


def load_mnist_as_npy(train_size):
    """[summary]

    [extended_summary]

    Args:
        train_size ([type]): [description]

    Returns:
        [type]: [description]
    """
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


def load_newsgroups_as_pickle(categories=None):
    """[summary]

    [extended_summary]

    Args:
        categories ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    newsgroups = fetch_20newsgroups(subset="all", categories=categories, remove=("headers", "footers", "quotes"),
                                    random_state=42)
    path_to_data = join(get_project_root(), "{}/{}.{}".format("data", "20newsgroups_data", "pickle"))
    path_to_labels = join(get_project_root(), "{}/{}.{}".format("data", "20newsgroups_labels", "npy"))
    path_to_label_names = join(get_project_root(), "{}/{}.{}".format("data", "20newsgroups_label_names", "pickle"))

    with open(path_to_data, "wb") as fp:
        pickle.dump(newsgroups["data"], fp)
    np.save(path_to_labels, newsgroups["target"])
    with open(path_to_label_names, "wb") as fp:
        pickle.dump(newsgroups["target_names"], fp)

    return path_to_data, path_to_labels, path_to_label_names


def _save_data_and_labels(data, labels, path_to_data, path_to_labels):
    """[summary]

    [extended_summary]

    Args:
        data ([type]): [description]
        labels ([type]): [description]
        path_to_data ([type]): [description]
        path_to_labels ([type]): [description]
    """
    if not isfile(path_to_data):
        np.save(path_to_data, data)
    if not isfile(path_to_labels):
        np.save(path_to_labels, labels)


def generate_unique_path(folder_name, extension, prefix=None):
    """[summary]

    [extended_summary]

    Args:
        folder_name ([type]): [description]
        extension ([type]): [description]
        prefix ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    root_folder = get_project_root()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    if prefix:
        return join(root_folder, "{}/{}_{}.{}".format(folder_name, prefix, timestamp, extension))
    return join(root_folder, "{}/{}.{}".format(folder_name, timestamp, extension))


def get_project_root():
    """[summary]

    [extended_summary]

    Returns:
        [type]: [description]
    """
    return dirname(dirname(realpath(__file__)))


def expand_parameter_to_linspace(param):
    """[summary]

    [extended_summary]

    Args:
        param ([type]): [description]

    Returns:
        [type]: [description]
    """
    if len(param) == 1:
        param = (param[0], param[0], 1)
    return np.linspace(*param)


def split_df_by_model(df):
    """[summary]

    [extended_summary]

    Args:
        df ([type]): [description]

    Returns:
        [type]: [description]
    """
    dfs = []
    for model_name, df_ in df.groupby("model_name"):
        df_ = df_.dropna(axis=1, how="all")
        df_ = df_.drop("model_name", axis=1)
        df_ = df_.reset_index(drop=True)
        df_.name = model_name
        dfs.append(df_)
    return dfs


def filter_optimized_results(df, err_param_name, score_name, is_higher_score_better):
    """[summary]

    [extended_summary]

    Args:
        df ([type]): [description]
        err_param_name ([type]): [description]
        score_name ([type]): [description]
        is_higher_score_better (bool): [description]

    Returns:
        [type]: [description]
    """
    if is_higher_score_better:
        df_ = df.loc[df.groupby(err_param_name, sort=False)[score_name].idxmax()].reset_index(drop=True)
    else:
        df_ = df.loc[df.groupby(err_param_name, sort=False)[score_name].idxmin()].reset_index(drop=True)
    df_.name = df.name
    return df_

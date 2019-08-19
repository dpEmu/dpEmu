import numpy as np
from mock import patch
from sklearn.datasets.base import Bunch
from dpemu import dataset_utils as utils


def mock_fetch_20newsgroups(subset="", categories=[], remove=(), random_state=1):
    lst = ['subject', 'object']
    target = np.array([0, 0, 1, 1, 0, 0, 0, 0, 1, 0])
    data = ["a", "b", "A", "B", "xy", "zs", "df", "io", "GD", "ga"]
    mock_newsgroups = Bunch(data=data, target=target, target_names=lst)
    return mock_newsgroups


def mock_fetch_openml():
    pass


def mock_load_digits():
    pass


def test_split_data_does_not_accept_wrong_train_size_parameters():
    data = np.arange(1, 101)
    labels = np.arange(1, 101).astype(str)

    ret_data, ret_labels = utils.split_data(data, labels, n_data=100)
    assert np.array_equal(ret_data, data)
    assert np.array_equal(ret_labels, labels)

    ret_data, ret_labels = utils.split_data(data, labels, n_data=0)
    assert np.array_equal(ret_data, data)
    assert np.array_equal(ret_labels, labels)

    ret_data, ret_labels = utils.split_data(data, labels, n_data=150)
    assert np.array_equal(ret_data, data)
    assert np.array_equal(ret_labels, labels)

    ret_data, ret_labels = utils.split_data(data, labels, n_data=-10)
    assert np.array_equal(ret_data, data)
    assert np.array_equal(ret_labels, labels)


@patch("dpemu.dataset_utils.fetch_20newsgroups", side_effect=mock_fetch_20newsgroups)
def test_split_data_returns_correct_values(mock_fetch_20newsgroups):
    ret_data, targets, target_names, descr = utils.load_newsgroups()

    assert np.array_equal(ret_data, ["a", "b", "A", "B", "xy", "zs", "df", "io", "GD", "ga"])
    assert np.array_equal(targets, np.array([0, 0, 1, 1, 0, 0, 0, 0, 1, 0]))
    assert np.array_equal(target_names, ['subject', 'object'])
    assert descr == "20newsgroups"

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


def mock_fetch_openml(string):
    rng = np.random.RandomState(1729)
    size = 100
    data = rng.rand(size, 784)
    target = rng.randint(low=0, high=10, size=size).astype(str)
    mock_mnist = Bunch(data=data, target=target)
    return mock_mnist


def mock_load_digits():
    rng = np.random.RandomState(1729)
    size = 10
    data = rng.randint(0, 256, size * 64).reshape(size, 64).astype(float)
    target = rng.randint(low=0, high=10, size=size)
    mock_digits = Bunch(data=data, target=target)
    return mock_digits


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


def test_split_data_returns_correct_size():
    data = np.arange(1, 101)
    labels = np.arange(1, 101).astype(str)
    ret_data, ret_labels = utils.split_data(data, labels, n_data=65)

    assert ret_data.shape[0] == 65
    assert ret_labels.shape[0] == 65


@patch("dpemu.dataset_utils.fetch_20newsgroups", side_effect=mock_fetch_20newsgroups)
def test_load_newsgroups(mock_fetch_20newsgroups):
    ret_data, targets, target_names, descr = utils.load_newsgroups()

    assert np.array_equal(ret_data, ["a", "b", "A", "B", "xy", "zs", "df", "io", "GD", "ga"])
    assert np.array_equal(targets, np.array([0, 0, 1, 1, 0, 0, 0, 0, 1, 0]))
    assert np.array_equal(target_names, ['subject', 'object'])
    assert descr == "20newsgroups"


@patch("dpemu.dataset_utils.fetch_20newsgroups", side_effect=mock_fetch_20newsgroups)
def test_load_newsgroups_with_wrong_n_categories(mock_fetch_20newsgroups):
    ret_data, targets, target_names, descr = utils.load_newsgroups(n_categories=21)

    assert np.array_equal(ret_data, ["a", "b", "A", "B", "xy", "zs", "df", "io", "GD", "ga"])
    assert np.array_equal(targets, np.array([0, 0, 1, 1, 0, 0, 0, 0, 1, 0]))
    assert np.array_equal(target_names, ['subject', 'object'])
    assert descr == "20newsgroups"


@patch("dpemu.dataset_utils.fetch_openml", side_effect=mock_fetch_openml)
def test_load_mnist(mock_fetch_openml):
    ret_data, ret_labels, _, descr = utils.load_mnist()
    test_rng = np.random.RandomState(1729)

    assert np.array_equal(ret_data, test_rng.rand(100, 784))
    assert np.array_equal(ret_labels, test_rng.randint(0, 10, size=100))
    assert descr == "MNIST"


@patch("dpemu.dataset_utils.fetch_openml", side_effect=mock_fetch_openml)
def test_load_fashion(mock_fetch_openml):
    ret_data, ret_labels, ret_label_names, descr = utils.load_fashion()
    test_rng = np.random.RandomState(1729)
    label_names = ["T-shirt", "Trouser", "Pullover", "Dress",
                   "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    assert np.array_equal(ret_data, test_rng.rand(100, 784))
    assert np.array_equal(ret_labels, test_rng.randint(0, 10, size=100))
    assert np.array_equal(ret_label_names, label_names)
    assert descr == "Fashion MNIST"


@patch("dpemu.dataset_utils.load_digits", side_effect=mock_load_digits)
def test_load_digits(mock_load_digits):
    ret_data, ret_labels, _, descr = utils.load_digits_(10)
    test_rng = np.random.RandomState(1729)

    test_data = test_rng.randint(0, 256, 640).reshape(10, 64).astype(float)
    assert np.allclose(ret_data, test_data)
    assert np.array_equal(ret_labels, test_rng.randint(low=0, high=10, size=10))
    assert descr == "Digits"

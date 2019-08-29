# MIT License
#
# Copyright (c) 2019 Tuomas Halvari, Juha Harviainen, Juha Mylläri, Antti Röyskö, Juuso Silvennoinen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import warnings
import os
import random as rn
from subprocess import Popen

import cv2
from numpy.random import RandomState
from pycocotools.coco import COCO
from sklearn.datasets import fetch_20newsgroups, fetch_openml, load_digits
from sklearn.model_selection import train_test_split

from dpemu.utils import get_project_root


random_state = RandomState(42)


def load_newsgroups(subset="all", n_categories=20):
    """Fetches the 20 newsgroups dataset and returns its desired subset.

    Args:
        subset (str, optional): If "test" then a smaller dataset is used instead of the full one. Defaults to "all".
        n_categories (int, optional): The number of categories to be included. Defaults to 20.

    Returns:
        tuple: The dataset, categories as integers, category names and the name of the dataset.
    """
    categories = [
        "alt.atheism",
        "comp.graphics",
        "sci.space",
        "rec.autos",
        "rec.sport.hockey",
        "rec.sport.baseball",
        "sci.electronics",
        "misc.forsale",
        "sci.crypt",
        "talk.politics.mideast",
        "sci.med",
        "comp.sys.mac.hardware",
        "comp.windows.x",
        "rec.motorcycles",
        "soc.religion.christian",
        "talk.politics.misc",
        "talk.religion.misc",
        "talk.politics.guns",
        "comp.sys.ibm.pc.hardware"
    ]
    if not 0 < n_categories < 21:
        n_categories = 20
    newsgroups = fetch_20newsgroups(subset=subset, categories=categories[:n_categories],
                                    remove=("headers", "footers", "quotes"), random_state=random_state)
    return newsgroups["data"], newsgroups["target"].astype(int), newsgroups["target_names"], "20newsgroups"


def __split_data(data, labels, n_data):
    """Returns a subset of a given size of the original data and labels.

    Args:
        data (list): Original data.
        labels (list): Original labels.
        n_data (int): Size of the subset.

    Returns:
        tuple: a subset of data and a subset of labels.
    """
    if 0 < n_data < data.shape[0]:
        data, _, labels, _ = train_test_split(data, labels, train_size=n_data, random_state=random_state)
    return data, labels


def load_digits_(n_data=1797):
    """Fetches the digits dataset and returns its desired subset.

    Args:
        n_data (int, optional): The size of the wanted subset. Defaults to 1797.

    Returns:
        tuple: The dataset, the labels of data points, the names of categories and the name of the dataset.
    """
    digits = load_digits()
    data, labels = __split_data(digits["data"], digits["target"], n_data)
    return data, labels, None, "Digits"


def load_mnist_unsplit(n_data=70000):
    """Fetches the MNIST dataset and returns its subset.

    Args:
        n_data (int, optional): The size of the wanted subset. Defaults to 70000.

    Returns:
        tuple: The dataset, the labels of data points, the names of categories and the name of the dataset.
    """
    mnist = fetch_openml("mnist_784")
    data, labels = __split_data(mnist["data"], mnist["target"].astype(int), n_data)
    return data, labels, None, "MNIST"


def load_mnist(reshape_to_28x28=False, integer_values=False):
    """Fetches the MNIST dataset and returns its desired subset.

    Args:
        reshape_to_28x28 (bool, optional): The data is reshaped to 28x28 images if true. Defaults to False.
        integer_values (bool, optional): The data is typecast to integers if true. Defaults to False.

    Returns:
        tuple: Training pixel data, training labels, test pixel data, test labels.
    """
    from contextlib import redirect_stderr
    warnings.simplefilter(action='ignore', category=FutureWarning)
    with redirect_stderr(open(os.devnull, 'w')):
        from keras.datasets.mnist import load_data as load_mnist_data

    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    if not reshape_to_28x28:
        x_train = x_train.reshape((-1, 28*28))
        x_test = x_test.reshape((-1, 28*28))
    if not integer_values:
        x_train = x_train.astype('float')
        y_train = y_train.astype('float')
        x_test = x_test.astype('float')
        y_train = y_train.astype('float')
    return x_train, y_train, x_test, y_test


def load_fashion(n_data=70000):
    """Fetches the fashion MNIST dataset and returns its desired subset.

    Args:
        n_data (int, optional): The size of the wanted subset. Defaults to 70000.

    Returns:
        tuple: The dataset, the labels of elements, the names of categories and the name of the dataset.
    """
    mnist = fetch_openml("Fashion-MNIST")
    data, labels = __split_data(mnist["data"], mnist["target"].astype(int), n_data)
    label_names = [
        "T-shirt",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot"
    ]
    return data, labels, label_names, "Fashion MNIST"


def load_coco_val_2017(n=5000, is_shuffled=False):
    """Fetches the COCO dataset and returns its desired subset.

    Args:
        n (int, optional): The size of the wanted subset. Defaults to 5000.
        is_shuffled (bool, optional): If true, then the chosen subset of the data will be shuffled. Defaults to False.

    Returns:
        tuple: The dataset, the labels of elements, the names of categories and the name of the dataset.
    """
    if n not in range(1, 5000):
        n = 5000
    img_folder = f"{get_project_root()}/data/val2017"
    if not os.path.isdir(img_folder):
        Popen(["./scripts/get_coco_dataset.sh"], cwd=get_project_root()).wait()

    coco = COCO(f"{get_project_root()}/data/annotations/instances_val2017.json")
    img_ids = coco.getImgIds()
    if is_shuffled:
        rn.shuffle(img_ids)
    img_ids = img_ids[:n]
    img_dicts = coco.loadImgs(img_ids)
    img_filenames = [img_dict["file_name"] for img_dict in img_dicts]
    imgs = [cv2.imread(os.path.join(img_folder, img_filename)) for img_filename in img_filenames]
    imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs]
    with open(f"{get_project_root()}/data/coco.names", "r") as fp:
        class_names = [line.strip() for line in fp.readlines()]

    return imgs, img_ids, class_names, img_filenames

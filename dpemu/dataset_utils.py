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

warnings.simplefilter(action='ignore', category=FutureWarning)
from keras.datasets.mnist import load_data as load_mnist_data

random_state = RandomState(42)


def load_newsgroups(subset="all", n_categories=20):
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


def split_data(data, labels, n_data):
    if 0 < n_data < data.shape[0]:
        data, _, labels, _ = train_test_split(data, labels, train_size=n_data, random_state=random_state)
    return data, labels


def load_digits_(n_data=1797):
    digits = load_digits()
    data, labels = split_data(digits["data"], digits["target"], n_data)
    return data, labels, None, "Digits"


def load_mnist(n_data=70000):
    mnist = fetch_openml("mnist_784")
    data, labels = split_data(mnist["data"], mnist["target"].astype(int), n_data)
    return data, labels, None, "MNIST"


def _load_mnist(reshape_to_28x28=False):
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    if reshape_to_28x28:
        x_train = x_train.reshape((-1, 28, 28))
        x_test = x_test.reshape((-1, 28, 28))
    return x_train, y_train, x_test, y_test

def load_fashion(n_data=70000):
    mnist = fetch_openml("Fashion-MNIST")
    data, labels = split_data(mnist["data"], mnist["target"].astype(int), n_data)
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

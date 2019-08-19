import os
import random as rn
import subprocess

import cv2
from numpy.random import RandomState
from pycocotools.coco import COCO
from sklearn.datasets import fetch_20newsgroups, fetch_openml, load_digits
from sklearn.model_selection import train_test_split

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
    img_folder = "data/val2017"
    if not os.path.isdir(img_folder):
        subprocess.call(["./scripts/get_coco_dataset.sh"])

    coco = COCO("data/annotations/instances_val2017.json")
    img_ids = coco.getImgIds()
    if is_shuffled:
        rn.shuffle(img_ids)
    img_ids = img_ids[:n]
    img_dicts = coco.loadImgs(img_ids)
    img_filenames = [img_dict["file_name"] for img_dict in img_dicts]
    imgs = [cv2.imread(os.path.join(img_folder, img_filename)) for img_filename in img_filenames]
    imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs]
    with open("data/coco.names", "r") as fp:
        class_names = [line.strip() for line in fp.readlines()]

    return imgs, img_ids, class_names, img_filenames

import os
import subprocess

import cv2
import pandas as pd
from numpy.random import RandomState
from pycocotools.coco import COCO
from sklearn.datasets import fetch_20newsgroups, fetch_openml, load_digits
from sklearn.model_selection import train_test_split

data_home = None
# data_home = "/wrk/users/thalvari/"
pd.set_option("display.expand_frame_repr", False)
random_state = RandomState(42)


def load_newsgroups(subset="all", n_categories=20):
    categories = [
        "alt.atheism",
        "comp.graphics",
        "sci.space",
        "rec.autos",
        "rec.sport.hockey",
        "sci.med",
        "rec.sport.baseball",
        "sci.electronics",
        "misc.forsale",
        "sci.crypt",
        "talk.politics.mideast",
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
                                    remove=("headers", "footers", "quotes"), random_state=random_state,
                                    data_home=data_home)
    return newsgroups["data"], newsgroups["target"].astype(int), newsgroups["target_names"], "20newsgroups"


def split_data(data, labels, n_data):
    if 0 < n_data < data.shape[0]:
        data, _, labels, _ = train_test_split(data, labels, train_size=n_data, random_state=random_state)
    return data, labels


def load_digits_(n_data=1797):
    mnist = load_digits()
    data, labels = split_data(mnist["data"], mnist["target"], n_data)
    return data, labels, None, "Digits"


def load_mnist(n_data=70000):
    mnist = fetch_openml("mnist_784", data_home=data_home)
    data, labels = split_data(mnist["data"], mnist["target"].astype(int), n_data)
    return data, labels, None, "MNIST"


def load_fashion(n_data=70000):
    mnist = fetch_openml("Fashion-MNIST", data_home=data_home)
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
        "Ankle boot",
    ]
    return data, labels, label_names, "Fashion MNIST"


def load_coco_val_2017(n=5000):
    if n not in range(1, 5000):
        n = 5000
    img_folder = "data/val2017"
    if not os.path.isdir(img_folder):
        subprocess.call(["./data/get_coco_dataset.sh"])
    path_to_yolov3_weights = "data/yolov3.weights"
    if not os.path.isfile(path_to_yolov3_weights):
        subprocess.call(["./data/get_yolov3.sh"])

    coco = COCO("data/annotations/instances_val2017.json")
    img_ids = sorted(coco.getImgIds())[:n]
    img_dicts = coco.loadImgs(img_ids)
    # imgs = [cv2.cvtColor(cv2.imread(os.path.join(img_folder, img_dict["file_name"])), cv2.COLOR_BGR2RGB) for img_dict in
    #         img_dicts]
    imgs = [cv2.imread(os.path.join(img_folder, img_dict["file_name"])) for img_dict in img_dicts]
    with open("data/coco.names", "r") as fp:
        class_names = [line.strip() for line in fp.readlines()]

    return imgs, img_ids, class_names

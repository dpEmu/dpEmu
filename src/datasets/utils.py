import pandas as pd
from numpy.random import RandomState

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
        "talk.politics.guns",
        "rec.sport.hockey",
        "comp.sys.ibm.pc.hardware",
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

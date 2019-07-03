import numpy as np
from numpy.random import RandomState

from sklearn.datasets import fetch_20newsgroups, fetch_openml, load_digits
from sklearn.model_selection import train_test_split

random_state = RandomState(42)
# data_home = "/wrk/users/thalvari/"
data_home = None


def load_newsgroups(n_categories=20):
    categories = [
        "alt.atheism",
        "talk.religion.misc",
        "comp.graphics",
        "sci.space",
        "comp.os.ms-windows.misc",
        "comp.sys.ibm.pc.hardware",
        "comp.sys.mac.hardware",
        "comp.windows.x",
        "misc.forsale",
        "rec.autos",
        "rec.motorcycles",
        "rec.sport.baseball",
        "rec.sport.hockey",
        "sci.crypt",
        "sci.electronics",
        "sci.med",
        "soc.religion.christian",
        "talk.politics.guns",
        "talk.politics.mideast",
        "talk.politics.misc",
    ]
    newsgroups = fetch_20newsgroups(subset="test", categories=categories[:n_categories],
                                    remove=("headers", "footers", "quotes"), random_state=random_state,
                                    data_home=data_home)
    return newsgroups["data"], np.array(newsgroups["target"].astype(int)), newsgroups["target_names"], "20newsgroups"


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

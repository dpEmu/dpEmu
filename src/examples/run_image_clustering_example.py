import sys
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from hdbscan import HDBSCAN
from numba.errors import NumbaDeprecationWarning, NumbaWarning
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.datasets import fetch_openml, load_digits
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

from src import runner_
from src.ml.utils import reduce_dimensions
from src.plotting.utils import visualize_scores, visualize_classes, visualize_interactive
from src.problemgenerator import array, copy, filters
from src.utils import split_data, split_df_by_model

warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaWarning)


class AbstractModel(ABC):

    def __init__(self):
        self.random_state = np.random.RandomState(1)

    @abstractmethod
    def get_fitted_model(self, reduced_data, model_params, n_classes):
        pass

    def run(self, data, model_params):
        labels = model_params["labels"]
        n_classes = len(np.unique(labels))

        reduced_data = reduce_dimensions(data, self.random_state)

        fitted_model = self.get_fitted_model(reduced_data, model_params, n_classes)

        return {
            "reduced_data": reduced_data,
            "ARI": round(adjusted_rand_score(labels, fitted_model.labels_), 3),
            "AMI": round(adjusted_mutual_info_score(labels, fitted_model.labels_, average_method="arithmetic"), 3),
        }


class KMeansModel(AbstractModel):

    def __init__(self):
        super().__init__()

    def get_fitted_model(self, reduced_data, model_params, n_classes):
        return KMeans(n_clusters=n_classes, random_state=self.random_state, n_jobs=1).fit(reduced_data)


class AgglomerativeModel(AbstractModel):

    def __init__(self):
        super().__init__()

    def get_fitted_model(self, reduced_data, model_params, n_classes):
        return AgglomerativeClustering(n_clusters=n_classes).fit(reduced_data)


class HDBSCANModel(AbstractModel):

    def __init__(self):
        super().__init__()

    def get_fitted_model(self, reduced_data, model_params, n_classes):
        return HDBSCAN(
            min_samples=1,
            min_cluster_size=model_params["min_cluster_size"],
            core_dist_n_jobs=1
        ).fit(reduced_data)


def load_digits_(n_data=1797):
    mnist = load_digits()
    data, labels = split_data(mnist["data"], mnist["target"], n_data)
    return data, labels, None, "Digits"


def load_mnist(n_data=70000):
    mnist = fetch_openml("mnist_784")
    # mnist = fetch_openml("mnist_784", data_home="/wrk/users/thalvari/")
    data, labels = split_data(mnist["data"], mnist["target"].astype(int), n_data)
    return data, labels, None, "MNIST"


def load_fashion(n_data=70000):
    mnist = fetch_openml("Fashion-MNIST")
    # mnist = fetch_openml("Fashion-MNIST", data_home="/wrk/users/thalvari/")
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


class ErrGen:
    def __init__(self, data):
        self.data = data

    def generate_error(self, params):
        data = deepcopy(self.data)
        data_node = array.Array(data.shape)
        root_node = copy.Copy(data_node)

        f = filters.GaussianNoise(params["mean"], params["std"])

        min_val = np.amin(self.data)
        max_val = np.amax(self.data)
        data_node.addfilter(filters.Min(filters.Max(f, filters.Constant(min_val)), filters.Constant(max_val)))

        return root_node.process(data, np.random.RandomState(42))


def visualize(dfs, label_names, dataset_name, data):
    visualize_interactive(
        dfs,
        "std",
        data,
        "tab10",
        "gray_r"
    )
    visualize_classes(
        dfs,
        label_names,
        "std",
        f"{dataset_name} (n={data.shape[0]}) classes with added gaussian noise"
    )
    visualize_scores(
        dfs,
        ["AMI", "ARI"],
        "std",
        f"{dataset_name} clustering scores with added gaussian noise"
    )
    plt.show()


def main(argv):
    if len(argv) == 3 and argv[1] == "digits":
        data, labels, label_names, dataset_name = load_digits_(int(argv[2]))
        std_steps = [0, 3, 6, 9, 12, 15]
    elif len(argv) == 3 and argv[1] == "mnist":
        data, labels, label_names, dataset_name = load_mnist(int(argv[2]))
        std_steps = [0, 51, 102, 153, 204, 255]
    elif len(argv) == 3 and argv[1] == "fashion":
        data, labels, label_names, dataset_name = load_fashion(int(argv[2]))
        std_steps = [0, 51, 102, 153, 204, 255]
    else:
        exit(0)

    n_data = data.shape[0]
    err_params_list = [{"mean": 0, "std": std} for std in std_steps]

    mcs_steps = map(int, n_data / np.array([12, 15, 20, 30, 55, 80, 140]))
    model_param_pairs = [
        (KMeansModel, [{"labels": labels}]),
        (AgglomerativeModel, [{"labels": labels}]),
        (HDBSCANModel, [{"min_cluster_size": mcs, "labels": labels} for mcs in mcs_steps]),
    ]

    df = runner_.run(ErrGen(data), err_params_list, model_param_pairs, True)
    dfs = split_df_by_model(df)

    for df in dfs:
        print(df.name)
        print(df.drop(columns=["labels", "reduced_data"]))

    visualize(dfs, label_names, dataset_name, data)


if __name__ == "__main__":
    main(sys.argv)

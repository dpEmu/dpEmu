import sys
import warnings
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from hdbscan import HDBSCAN
from numba.errors import NumbaDeprecationWarning, NumbaWarning
from numpy.random import RandomState
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

from src import runner_
from src.datasets.utils import load_digits_, load_mnist, load_fashion
from src.ml.utils import reduce_dimensions

from src.plotting.utils import visualize_scores, visualize_classes, visualize_interactive_plot, print_results
from src.problemgenerator.array import Array
from src.problemgenerator.copy import Copy
from src.problemgenerator.filters import GaussianNoise, Min, Max, Constant

warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaWarning)


class AbstractModel(ABC):

    def __init__(self):
        self.random_state = RandomState(42)

    @abstractmethod
    def get_fitted_model(self, reduced_data, model_params, n_classes):
        pass

    def run(self, _, data, model_params):
        labels = model_params["labels"]
        n_classes = len(np.unique(labels))

        reduced_data = reduce_dimensions(data, self.random_state)

        fitted_model = self.get_fitted_model(reduced_data, model_params, n_classes)

        return {
            "AMI": round(adjusted_mutual_info_score(labels, fitted_model.labels_, average_method="arithmetic"), 3),
            "ARI": round(adjusted_rand_score(labels, fitted_model.labels_), 3),
            "reduced_data": reduced_data,
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


class ErrGen:
    def __init__(self):
        self.random_state = RandomState(42)

    def generate_error(self, data, params):
        data = np.array(data)

        data_node = Array(data.shape)
        root_node = Copy(data_node)

        f = GaussianNoise(params["mean"], params["std"])

        min_val = np.amin(data)
        max_val = np.amax(data)
        data_node.addfilter(Min(Max(f, Constant(min_val)), Constant(max_val)))

        return root_node.process(data, self.random_state)


def visualize(df, label_names, dataset_name, data):
    visualize_scores(df, ["AMI", "ARI"], "std", f"{dataset_name} clustering scores with added gaussian noise")
    visualize_classes(df, label_names, "std", "reduced_data", "labels",
                      f"{dataset_name} (n={data.shape[0]}) classes with added gaussian noise")
    visualize_interactive_plot(df, "std", data, "tab10", "gray_r")  # Remember to enable runner's interactive mode
    plt.show()


def main(argv):
    if len(argv) == 3 and argv[1] == "digits":
        data, labels, label_names, dataset_name = load_digits_(int(argv[2]))
    elif len(argv) == 3 and argv[1] == "mnist":
        data, labels, label_names, dataset_name = load_mnist(int(argv[2]))
    elif len(argv) == 3 and argv[1] == "fashion":
        data, labels, label_names, dataset_name = load_fashion(int(argv[2]))
    else:
        exit(0)

    max_val = np.amax(data)
    std_steps = np.linspace(0, max_val, num=8)
    err_params_list = [{"mean": 0, "std": std} for std in std_steps]

    n_data = data.shape[0]
    divs = [12, 15, 20, 30, 55, 80, 140]
    mcs_steps = [round(n_data / div) for div in divs]
    model_params_dict_list = [
        {"model": KMeansModel, "params_list": [{"labels": labels}]},
        {"model": AgglomerativeModel, "params_list": [{"labels": labels}]},
        {"model": HDBSCANModel, "params_list": [{"min_cluster_size": mcs, "labels": labels} for mcs in mcs_steps]},
    ]

    df = runner_.run(None, data, ErrGen, err_params_list, model_params_dict_list, True)

    print_results(df, ["labels", "reduced_data", "err_test_data"])
    visualize(df, label_names, dataset_name, data)


if __name__ == "__main__":
    main(sys.argv)

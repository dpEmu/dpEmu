import random as rn
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from hdbscan import HDBSCAN
from numba.errors import NumbaDeprecationWarning, NumbaWarning
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.datasets import fetch_openml, load_digits
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.model_selection import train_test_split
from umap import UMAP

from src import runner_
from src.problemgenerator import array, copy, filters
from src.utils import generate_unique_path

warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaWarning)


class AbstractModel(ABC):

    def __init__(self):
        rn.seed(42)
        np.random.seed(42)

    @abstractmethod
    def get_fitted_model(self, reduced_data, labels, model_params):
        pass

    def run(self, data, model_params):
        labels = model_params["labels"]
        pca_limit = 30

        reduced_data = data
        if reduced_data.shape[1] > pca_limit:
            reduced_data = PCA(n_components=pca_limit, random_state=42).fit_transform(reduced_data)
        reduced_data = UMAP(n_neighbors=30, min_dist=0.0, random_state=42).fit_transform(reduced_data)

        fitted_model = self.get_fitted_model(reduced_data, labels, model_params)

        return {
            "reduced_data": reduced_data,
            "ARI": round(adjusted_rand_score(labels, fitted_model.labels_), 3),
            "AMI": round(adjusted_mutual_info_score(labels, fitted_model.labels_, average_method="arithmetic"), 3),
        }


class KMeansModel(AbstractModel):

    def __init__(self):
        super().__init__()

    def get_fitted_model(self, reduced_data, labels, model_params):
        n_classes = len(np.unique(labels))
        return KMeans(n_clusters=n_classes, random_state=42).fit(reduced_data)


class AgglomerativeModel(AbstractModel):

    def __init__(self):
        super().__init__()

    def get_fitted_model(self, reduced_data, labels, model_params):
        n_classes = len(np.unique(labels))
        return AgglomerativeClustering(n_clusters=n_classes).fit(reduced_data)


class HDBSCANModel(AbstractModel):

    def __init__(self):
        super().__init__()

    def get_fitted_model(self, reduced_data, labels, model_params):
        return HDBSCAN(min_samples=10, min_cluster_size=model_params["min_cluster_size"]).fit(reduced_data)


def split_data(data, labels, train_size):
    if train_size < data.shape[0]:
        data, _, labels, _ = train_test_split(data, labels, train_size=train_size, random_state=42)
    return data, labels


def load_digits_(train_size=1797):
    mnist = load_digits()
    data, labels = split_data(mnist["data"], mnist["target"], train_size)
    return data, labels, None


def load_mnist(train_size=70000):
    mnist = fetch_openml("mnist_784")
    # mnist = fetch_openml("mnist_784", data_home="/wrk/users/thalvari/")
    data, labels = split_data(mnist["data"], mnist["target"].astype(int), train_size)
    return data, labels, None


def load_fashion(train_size=70000):
    mnist = fetch_openml("Fashion-MNIST")
    # mnist = fetch_openml("Fashion-MNIST", data_home="/wrk/users/thalvari/")
    data, labels = split_data(mnist["data"], mnist["target"].astype(int), train_size)
    return data, labels, [
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


class ErrGen:
    def __init__(self, data):
        self.data = data

    def generate_error(self, params):
        data = deepcopy(self.data)
        data_node = array.Array(data.shape)
        root_node = copy.Copy(data_node)

        data_node.addfilter(filters.GaussianNoise(params["mean"], params["std"]))

        return root_node.process(data, np.random.RandomState(seed=42))


def visualize_scores(dfs):
    xlabel = "std"
    scores = ["AMI", "ARI"]

    def inherit_name(df_new, df_old):
        df_new.name = df_old.name
        return df_new

    dfs_with_mcs = [df for df in dfs if "min_cluster_size" in df]
    dfs = [df for df in dfs if "min_cluster_size" not in df]
    for df_with_mcs in dfs_with_mcs:
        dfs.extend([inherit_name(df, df_with_mcs) for _, df in df_with_mcs.groupby("min_cluster_size")])

    plt.clf()

    _, axs = plt.subplots(1, 2, figsize=(8, 4))
    for i, ax in enumerate(axs.ravel()):
        for df in dfs:
            if "min_cluster_size" in df:
                ax.plot(df[xlabel], df[scores[i]], label=df.name + str(df["min_cluster_size"].values[0]))
            else:
                ax.plot(df[xlabel], df[scores[i]], label=df.name)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(scores[i])
            ax.set_xlim([0, df[xlabel].max()])
            ax.set_ylim([0, 1])
            ax.legend()

    plt.subplots_adjust(wspace=.25)
    plt.suptitle("Clustering scores with added gaussian noise")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    path_to_plot = generate_unique_path("out", "png")
    plt.savefig(path_to_plot)


def visualize_classes(dfs, label_names):
    def get_lims(data):
        return data[:, 0].min() - 1, data[:, 0].max() + 1, data[:, 1].min() - 1, data[:, 1].max() + 1

    df = dfs[0]
    if "min_cluster_size" in df:
        df = list(df.groupby("min_cluster_size"))[0][1].reset_index(drop=True)
    labels = df["labels"][0]

    plt.clf()
    fig, axs = plt.subplots(2, 3, figsize=(8, 5))
    for i, ax in enumerate(axs.ravel()):
        reduced_data = df["reduced_data"][i]
        x_min, x_max, y_min, y_max = get_lims(reduced_data)
        sc = ax.scatter(*reduced_data.T, c=labels, cmap="tab10", marker=".", s=40)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title("std=" + str(df["std"][i]))
        ax.set_xticks([])
        ax.set_yticks([])
    n_data = df["reduced_data"].values[0].shape[0]
    fig.suptitle(f"MNIST (n={n_data}) classes with added gaussian noise")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    cbar = fig.colorbar(sc, ax=axs, boundaries=np.arange(11) - 0.5, ticks=np.arange(10), use_gridspec=True)
    if label_names:
        cbar.ax.yaxis.set_ticklabels(label_names)

    path_to_plot = generate_unique_path("out", "png")
    fig.savefig(path_to_plot)


def visualize(dfs, label_names):
    visualize_classes(dfs, label_names)
    visualize_scores(dfs)


def main():
    n_data = 5000
    data, labels, label_names = load_digits_(n_data)
    # data, labels, label_names = load_mnist(n_data)
    # data, labels, label_names = load_fashion(n_data)
    n_data = data.shape[0]

    std_steps = [0, 3, 6, 9, 12, 15]  # For digits
    # std_steps = [0, 51, 102, 153, 204, 255]  # For mnist and fashion
    err_params_list = [{"mean": 0, "std": std} for std in std_steps]

    mcs_steps = map(int, [n_data / 75, n_data / 50, n_data / 25])
    model_param_pairs = [
        (KMeansModel(), [{"labels": labels}]),
        (AgglomerativeModel(), [{"labels": labels}]),
        (HDBSCANModel(), [{"labels": labels, "min_cluster_size": mcs} for mcs in mcs_steps]),
    ]

    dfs = runner_.run(ErrGen(data), err_params_list, model_param_pairs)

    for df in dfs:
        print(df.name)
        print(df.drop(columns=["labels", "reduced_data"]))

    visualize(dfs, label_names)


if __name__ == "__main__":
    main()

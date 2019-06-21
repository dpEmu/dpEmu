import random as rn
from abc import ABC, abstractmethod
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from umap import UMAP

from src import runner
from src.problemgenerator import array, copy, filters
from src.utils import generate_unique_path


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


class HDBSCAN500Model(AbstractModel):

    def __init__(self):
        super().__init__()

    def get_fitted_model(self, reduced_data, labels, model_params):
        return HDBSCAN(min_samples=10, min_cluster_size=500).fit(reduced_data)


class HDBSCAN1000Model(AbstractModel):

    def __init__(self):
        super().__init__()

    def get_fitted_model(self, reduced_data, labels, model_params):
        return HDBSCAN(min_samples=10, min_cluster_size=1000).fit(reduced_data)


class HDBSCAN1500Model(AbstractModel):

    def __init__(self):
        super().__init__()

    def get_fitted_model(self, reduced_data, labels, model_params):
        return HDBSCAN(min_samples=10, min_cluster_size=1500).fit(reduced_data)


def load_mnist(train_size=70000):
    mnist = fetch_openml("mnist_784")
    # mnist = fetch_openml("mnist_784", data_home="/wrk/users/thalvari/")
    # mnist = fetch_openml("Fashion-MNIST")
    # mnist = fetch_openml("Fashion-MNIST", data_home="/wrk/users/thalvari/")
    if train_size == mnist["data"].shape[0]:
        data = mnist["data"]
        labels = mnist["target"].astype(int)
    else:
        data, _, labels, _ = train_test_split(mnist["data"], mnist["target"].astype(int), train_size=train_size,
                                              random_state=42)
    return data, labels


class ErrGen:
    def __init__(self, data):
        self.data = data

    def generate_error(self, params):
        data = deepcopy(self.data)
        data_node = array.Array(data.shape)
        root_node = copy.Copy(data_node)

        data_node.addfilter(filters.GaussianNoise(params["mean"], params["std"]))

        return root_node.process(data, np.random.RandomState(seed=42))


class ParamSelector:
    def __init__(self, params):
        self.params = params

    def next(self):
        return self.params

    def analyze(self, res):
        self.params = None


def visualize_scores(dfs):
    xlabel = "std"

    plt.clf()
    plt.figure(figsize=(8, 4))
    plt.subplot(121)
    for df in dfs:
        plt.plot(df[xlabel], df["AMI"], label=df.name)
        plt.xlabel(xlabel)
        plt.ylabel("AMI")
        plt.xlim([0, 255])
        plt.ylim([0, 1])
        plt.legend()
    plt.subplot(122)
    for df in dfs:
        plt.plot(df[xlabel], df["ARI"], label=df.name)
        plt.xlabel(xlabel)
        plt.ylabel("ARI")
        plt.xlim([0, 255])
        plt.ylim([0, 1])
        plt.legend()
    plt.subplots_adjust(wspace=.25)
    plt.suptitle("Clustering scores with added gaussian noise")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    path_to_plot = generate_unique_path("out", "png")
    plt.savefig(path_to_plot)


def visualize_classes(dfs):
    def get_lims(data):
        return data[:, 0].min() - 1, data[:, 0].max() + 1, data[:, 1].min() - 1, data[:, 1].max() + 1

    df = dfs[0]
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
    fig.suptitle("MNIST classes with added gaussian noise")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.colorbar(sc, ax=axs, boundaries=np.arange(11) - 0.5, ticks=np.arange(10), use_gridspec=True)

    path_to_plot = generate_unique_path("out", "png")
    fig.savefig(path_to_plot)


def visualize(dfs):
    visualize_scores(dfs)
    visualize_classes(dfs)


def main():
    data, labels = load_mnist(500)
    # data, labels = load_mnist()

    err_gen = ErrGen(data)
    steps = [0, 51, 102, 153, 204, 255]
    dfs = []

    model_param_pairs = [
        (KMeansModel(), ParamSelector([({"mean": 0, "std": std}, {"labels": labels}) for std in steps])),
        (AgglomerativeModel(), ParamSelector([({"mean": 0, "std": std}, {"labels": labels}) for std in steps])),
        (HDBSCAN500Model(), ParamSelector([({"mean": 0, "std": std}, {"labels": labels}) for std in steps])),
        (HDBSCAN1000Model(), ParamSelector([({"mean": 0, "std": std}, {"labels": labels}) for std in steps])),
        (HDBSCAN1500Model(), ParamSelector([({"mean": 0, "std": std}, {"labels": labels}) for std in steps])),
    ]

    for model_param_pair in tqdm(model_param_pairs):
        df = runner.run(model_param_pair[0], err_gen, model_param_pair[1])
        df.name = model_param_pair[0].__class__.__name__.replace("Model", "")
        dfs.append(df)

    for df in dfs:
        print(df.drop(columns=["batch", "labels", "reduced_data"]))

    visualize(dfs)


if __name__ == "__main__":
    main()

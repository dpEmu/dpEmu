import io
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.metrics import v_measure_score, adjusted_rand_score, adjusted_mutual_info_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.random_projection import johnson_lindenstrauss_min_dim, SparseRandomProjection
from umap import UMAP

from src import runner
from src.problemgenerator import array, copy, filters
from src.utils import generate_unique_path


class Model:

    def __init__(self):
        np.random.seed(42)

    @staticmethod
    def __plot_to_img():
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        byte_img = buf.read()
        byte_img = io.BytesIO(byte_img)
        return Image.open(byte_img)

    def __get_classes_img(self, reduced_data, labels):
        x_min, x_max, y_min, y_max = self.__get_lims(reduced_data)

        plt.figure(1)
        plt.clf()
        plt.scatter(*reduced_data.T, c=labels, cmap="tab10", marker=".", s=40)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        return self.__plot_to_img()

    def __get_clusters_img(self, fitted_model, reduced_data):
        x_min, x_max, y_min, y_max = self.__get_lims(reduced_data)
        step = .01
        x, y = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))

        predicted_clusters = fitted_model.predict(np.c_[x.ravel(), y.ravel()])
        predicted_clusters = predicted_clusters.reshape(x.shape)
        centroids = fitted_model.cluster_centers_

        plt.figure(1)
        plt.clf()
        plt.imshow(predicted_clusters, extent=(x_min, x_max, y_min, y_max), cmap="tab10", aspect="auto", origin="lower")
        plt.scatter(*reduced_data.T, c="k", marker=".", s=40)
        plt.scatter(*centroids.T, marker="X", s=300, color="w", zorder=10)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        return self.__plot_to_img()

    @staticmethod
    def __get_lims(reduced_data):
        return (
            reduced_data[:, 0].min() - 1,
            reduced_data[:, 0].max() + 1,
            reduced_data[:, 1].min() - 1,
            reduced_data[:, 1].max() + 1
        )

    def run(self, data, model_params):
        if len(data.shape) == 3:
            data = data.reshape(self.data.shape[:-2] + (-1,))
        labels = model_params["labels"]

        n_classes = len(np.unique(labels))
        jl_limit = johnson_lindenstrauss_min_dim(n_samples=data.shape[0], eps=.3)
        pca_limit = 50
        sample_size = 300

        reduced_data = data
        if reduced_data.shape[1] > jl_limit and reduced_data.shape[1] > pca_limit:
            reduced_data = SparseRandomProjection(n_components=jl_limit, random_state=42).fit_transform(reduced_data)
        if reduced_data.shape[1] > pca_limit:
            reduced_data = PCA(n_components=pca_limit, random_state=42).fit_transform(reduced_data)
        reduced_data = UMAP(random_state=42).fit_transform(reduced_data)

        fitted_model = KMeans(n_clusters=n_classes, random_state=42).fit(reduced_data)

        return {
            "classes_img": self.__get_classes_img(reduced_data, labels),
            "clusters_img": self.__get_clusters_img(fitted_model, reduced_data),
            "v-meas": round(v_measure_score(labels, fitted_model.labels_), 3),
            "ARI": round(adjusted_rand_score(labels, fitted_model.labels_), 3),
            "AMI": round(adjusted_mutual_info_score(labels, fitted_model.labels_, average_method="arithmetic"), 3),
            "silhouette": round(silhouette_score(reduced_data, fitted_model.labels_, sample_size=sample_size), 3),
        }


def load_mnist(train_size=1000):
    mnist = fetch_openml("mnist_784")
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


def visualize(df):
    xlabel = "std"

    plt.figure(1)
    plt.clf()
    plt.plot(df[xlabel], df["v-meas"], label="v-meas")
    plt.plot(df[xlabel], df["AMI"], label="AMI")
    plt.plot(df[xlabel], df["ARI"], label="ARI")
    plt.plot(df[xlabel], df["silhouette"], label="silhouette")

    plt.legend()
    plt.xlim([0, 255])
    plt.ylim([0, 1])
    plt.xlabel(xlabel)
    plt.ylabel("score")
    plt.title("Gaussian noise")
    plt.tight_layout()
    path_to_plot = generate_unique_path("out", "png")
    plt.savefig(path_to_plot)

    for img_name in ["classes_img", "clusters_img"]:
        plt.figure(1)
        plt.clf()
        _, axs = plt.subplots(2, 3)
        if img_name == "classes_img":
            plt.suptitle("MNIST classes")
        else:
            plt.suptitle("KMeans clusters")
        for i, ax in enumerate(axs.ravel()):
            ax.imshow(df[img_name][i])
            ax.set_title("std " + str(df["std"][i]))
            ax.set_xticks([])
            ax.set_yticks([])
        plt.tight_layout()
        path_to_plot = generate_unique_path("out", "png")
        plt.savefig(path_to_plot)


def main():
    data, labels = load_mnist(500)

    model = Model()
    err_gen = ErrGen(data)

    param_selector = ParamSelector([({"mean": 0, "std": std}, {"labels": labels}) for std in range(0, 256, 51)])

    df = runner.run(model, err_gen, param_selector)

    print(df)
    visualize(df)


if __name__ == "__main__":
    main()

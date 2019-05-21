import io

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.metrics import v_measure_score, adjusted_rand_score, adjusted_mutual_info_score, silhouette_score
from umap import UMAP


class ClusteringAnalyzer:

    def __init__(self):
        self.seed = 42
        np.random.seed(self.seed)

    def analyze(self, data, labels):
        n_classes = len(np.unique(labels))
        reduced_data = UMAP(random_state=self.seed).fit_transform(data)
        estimator = KMeans(n_clusters=n_classes, random_state=self.seed).fit(reduced_data)
        return (
            self.__get_scores(reduced_data, estimator, labels),
            self.__generate_classes_img(reduced_data, labels),
            self.__generate_clusters_img(reduced_data, estimator)
        )

    @staticmethod
    def __get_scores(data, estimator, labels):
        sample_size = 300
        return {
            "v-meas": round(v_measure_score(labels, estimator.labels_), 3),
            "ARI": round(adjusted_rand_score(labels, estimator.labels_), 3),
            "AMI": round(adjusted_mutual_info_score(labels, estimator.labels_, average_method="arithmetic"), 3),
            "silhouette": round(silhouette_score(data, estimator.labels_, sample_size=sample_size), 3),
        }

    @staticmethod
    def __get_lims(data):
        return data[:, 0].min() - 1, data[:, 0].max() + 1, data[:, 1].min() - 1, data[:, 1].max() + 1

    def __generate_classes_img(self, data, labels):
        x_min, x_max, y_min, y_max = self.__get_lims(data)

        plt.figure(1)
        plt.clf()
        plt.scatter(*data.T, c=labels, cmap="tab10", marker=".", s=20)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(())
        plt.yticks(())
        plt.tight_layout()

        return self.__plt_to_img()

    def __generate_clusters_img(self, data, estimator):
        x_min, x_max, y_min, y_max = self.__get_lims(data)
        h = .02
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])
        z = z.reshape(xx.shape)
        centroids = estimator.cluster_centers_

        plt.figure(1)
        plt.clf()
        plt.imshow(z, interpolation="nearest", extent=(xx.min(), xx.max(), yy.min(), yy.max()), cmap="tab10",
                   aspect="auto", origin="lower")
        plt.scatter(*data.T, c="k", marker=".", s=20)
        plt.scatter(*centroids.T, marker="X", s=300, linewidths=1, color="w", zorder=10)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(())
        plt.yticks(())
        plt.tight_layout()

        return self.__plt_to_img()

    @staticmethod
    def __plt_to_img():
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        byte_img = buf.read()
        byte_img = io.BytesIO(byte_img)
        return Image.open(byte_img)


def digits_example():
    digits = load_digits()
    data = digits.data
    labels = digits.target

    print(data.shape, type(data))
    print(labels.shape, type(labels))

    cluster_analyzer = ClusteringAnalyzer()
    scores, classes_img, clusters_img = cluster_analyzer.analyze(data, labels)

    print(type(classes_img))
    print(scores)
    classes_img.show()
    clusters_img.show()


if __name__ == "__main__":
    digits_example()

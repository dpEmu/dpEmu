import json
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.externals.joblib import load
from sklearn.metrics import v_measure_score, adjusted_rand_score, adjusted_mutual_info_score, silhouette_score


class ClusteringAnalyzer:

    def __init__(self, params):
        self.reduced_data = np.load(params["path_to_reduced_data"])
        self.labels = np.load(params["path_to_labels"])
        self.fitted_model = load(params["path_to_fitted_model"])
        self.path_to_classes_img = params["path_to_classes_img"]
        self.path_to_clusters_img = params["path_to_clusters_img"]
        self.path_to_scores = params["path_to_scores"]
        np.random.seed(42)

    def analyze(self, ):
        self.__save_classes_img()
        self.__save_clusters_img()
        scores = self.__get_scores()

        with open(self.path_to_scores, "w") as fp:
            json.dump(scores, fp)

    def __get_scores(self):
        sample_size = 300
        scores = {
            "v-meas": v_measure_score(self.labels, self.fitted_model.labels_),
            "ARI": adjusted_rand_score(self.labels, self.fitted_model.labels_),
            "AMI": adjusted_mutual_info_score(self.labels, self.fitted_model.labels_, average_method="arithmetic"),
            "silhouette": silhouette_score(self.reduced_data, self.fitted_model.labels_, sample_size=sample_size),
        }
        return {k: str(round(v, 3)) for k, v in scores.items()}

    def __save_classes_img(self):
        x_min, x_max, y_min, y_max = self.__get_lims()

        plt.figure(1)
        plt.clf()
        plt.scatter(*self.reduced_data.T, c=self.labels, cmap="tab10", marker=".", s=20)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(self.path_to_classes_img)

    def __save_clusters_img(self):
        x_min, x_max, y_min, y_max = self.__get_lims()
        step = .01
        x, y = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))

        predicted_clusters = self.fitted_model.predict(np.c_[x.ravel(), y.ravel()])
        predicted_clusters = predicted_clusters.reshape(x.shape)
        centroids = self.fitted_model.cluster_centers_

        plt.figure(1)
        plt.clf()
        plt.imshow(predicted_clusters, extent=(x_min, x_max, y_min, y_max), cmap="tab10", aspect="auto", origin="lower")
        plt.scatter(*self.reduced_data.T, c="k", marker=".", s=20)
        plt.scatter(*centroids.T, marker="X", s=300, color="w", zorder=10)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(self.path_to_clusters_img)

    def __get_lims(self):
        return (
            self.reduced_data[:, 0].min() - 1,
            self.reduced_data[:, 0].max() + 1,
            self.reduced_data[:, 1].min() - 1,
            self.reduced_data[:, 1].max() + 1
        )


def main(argv):
    with open(argv[1], "r") as fp:
        params = json.load(fp)
    clustering_analyzer = ClusteringAnalyzer(params)
    clustering_analyzer.analyze()


if __name__ == "__main__":
    main(sys.argv)

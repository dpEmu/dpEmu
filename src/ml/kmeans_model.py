import json
import sys

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.externals.joblib import dump
from sklearn.random_projection import johnson_lindenstrauss_min_dim, SparseRandomProjection
from umap import UMAP


class ReducedKMeans:

    def __init__(self, params):
        self.data = np.load(params["path_to_data"])
        self.labels = np.load(params["path_to_labels"])
        self.path_to_reduced_data = params["path_to_reduced_data"]
        self.path_to_fitted_model = params["path_to_fitted_model"]
        self.seed = 42
        np.random.seed(self.seed)

    def reduce_and_fit_data(self):
        n_features = self.data.shape[1]
        n_classes = len(np.unique(self.labels))
        jl_limit = johnson_lindenstrauss_min_dim(n_samples=self.data.shape[0], eps=.3)
        reduced_data = self.data

        if n_features > jl_limit and n_features > 100:
            reduced_data = SparseRandomProjection(n_components=jl_limit, random_state=self.seed).fit_transform(
                reduced_data)

        if n_features > 100:
            reduced_data = PCA(n_components=100, random_state=self.seed).fit_transform(reduced_data)

        reduced_data = UMAP(random_state=self.seed).fit_transform(reduced_data)

        fitted_model = KMeans(n_clusters=n_classes, random_state=self.seed).fit(reduced_data)

        np.save(self.path_to_reduced_data, reduced_data)
        dump(fitted_model, self.path_to_fitted_model)


def main(argv):
    with open(argv[1], "r") as file:
        params = json.load(file)
    reduced_kmeans = ReducedKMeans(params)
    reduced_kmeans.reduce_and_fit_data()


if __name__ == "__main__":
    main(sys.argv)

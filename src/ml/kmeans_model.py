import sys

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.externals.joblib import dump
from sklearn.random_projection import johnson_lindenstrauss_min_dim, SparseRandomProjection
from umap.umap_ import UMAP


class ReducedKMeans:

    def __init__(self, paths):
        self.data = np.load(paths[0])
        if len(self.data.shape) == 3:
            self.data = self.data.reshape(self.data.shape[:-2] + (-1,))
        self.labels = np.load(paths[1])
        self.path_to_reduced_data = paths[2]
        self.path_to_fitted_model = paths[3]
        self.seed = 42
        np.random.seed(self.seed)

    def reduce_and_fit_data(self):
        n_classes = len(np.unique(self.labels))
        jl_limit = johnson_lindenstrauss_min_dim(n_samples=self.data.shape[0], eps=.3)
        pca_limit = 50
        reduced_data = self.data

        if reduced_data.shape[1] > jl_limit and reduced_data.shape[1] > pca_limit:
            reduced_data = SparseRandomProjection(n_components=jl_limit, random_state=self.seed).fit_transform(
                reduced_data)

        if reduced_data.shape[1] > pca_limit:
            reduced_data = PCA(n_components=pca_limit, random_state=self.seed).fit_transform(reduced_data)

        reduced_data = UMAP(random_state=self.seed).fit_transform(reduced_data)

        fitted_model = KMeans(n_clusters=n_classes, random_state=self.seed).fit(reduced_data)

        np.save(self.path_to_reduced_data, reduced_data)
        dump(fitted_model, self.path_to_fitted_model)


def main(argv):
    reduced_kmeans = ReducedKMeans([argv[1], argv[2], argv[3], argv[4]])
    reduced_kmeans.reduce_and_fit_data()


if __name__ == "__main__":
    main(sys.argv)

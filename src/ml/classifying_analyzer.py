import json
import pickle
import sys

import numpy as np
from joblib import load
from scipy.sparse import load_npz
from sklearn.model_selection import train_test_split


class ClusteringAnalyzer:

    def __init__(self, paths):
        np.random.seed(42)
        self.vectorized_data = load_npz(paths[0])
        with open(paths[1], "rb") as file:
            self.labels = pickle.load(file)
        self.fitted_clf = load(paths[2])
        self.path_to_scores = paths[3]
        self.path_to_best_clf_params = paths[4]

    def analyze(self):
        vectorized_train_data, vectorized_test_data, train_labels, test_labels = train_test_split(
            self.vectorized_data,
            self.labels,
            test_size=.2,
            random_state=42
        )

        scores = self.__get_scores(vectorized_train_data, vectorized_test_data, train_labels, test_labels)

        with open(self.path_to_scores, "w") as fp:
            json.dump(scores, fp)

        with open(self.path_to_best_clf_params, "w") as fp:
            json.dump(self.fitted_clf.best_params_, fp)

    def __get_scores(self, vectorized_train_data, vectorized_test_data, train_labels, test_labels):
        scores = {
            "train_set_mean_accuracy": self.fitted_clf.score(vectorized_train_data, train_labels),
            "test_set_mean_accuracy": self.fitted_clf.score(vectorized_test_data, test_labels),
        }
        return {k: str(round(v, 3)) for k, v in scores.items()}


def main(argv):
    classifying_analyzer = ClusteringAnalyzer([argv[1], argv[2], argv[3], argv[4], argv[5]])
    classifying_analyzer.analyze()


if __name__ == "__main__":
    main(sys.argv)

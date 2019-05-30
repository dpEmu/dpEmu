import json
import sys

import numpy as np
from sklearn.externals.joblib import load
from sklearn.model_selection import train_test_split


class ClusteringAnalyzer:

    def __init__(self, paths):
        self.data = np.load(paths[0])
        self.labels = np.load(paths[1])
        self.fitted_model = load(paths[2])
        self.path_to_scores = paths[3]
        self.path_to_best_params = paths[4]
        np.random.seed(42)

    def analyze(self):
        train_data, test_data, train_labels, test_labels = train_test_split(self.data, self.labels, test_size=.2,
                                                                            random_state=42)
        scores = self.__get_scores(test_data, test_labels)
        best_params = self.__get_best_params()

        with open(self.path_to_scores, "w") as fp:
            json.dump(scores, fp)

        with open(self.path_to_best_params, "w") as fp:
            json.dump(best_params, fp)

    def __get_scores(self, test_data, test_labels):
        scores = {
            "mean_accuracy": self.fitted_model.score(test_data, test_labels),
        }
        return {k: str(round(v, 3)) for k, v in scores.items()}

    def __get_best_params(self):
        return {k: str(round(v, 3)) for k, v in self.fitted_model.best_params_.items()}


def main(argv):
    classifying_analyzer = ClusteringAnalyzer([argv[1], argv[2], argv[3], argv[4], argv[5]])
    classifying_analyzer.analyze()


if __name__ == "__main__":
    main(sys.argv)

import json
import pickle
import sys

import numpy as np
from joblib import load
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


class ClassificationAnalyzer:

    def __init__(self, paths):
        np.random.seed(42)
        with open(paths[0], "rb") as fp:
            self.data = pickle.load(fp)
        self.labels = np.load(paths[1])
        with open(paths[2], "rb") as fp:
            self.label_names = pickle.load(fp)
        self.fitted_clf = load(paths[3])
        self.path_to_best_clf_params = paths[4]
        self.path_to_scores = paths[5]
        self.path_to_confusion_matrix_img = paths[6]

    def __build_cm_image(self, cm):
        # Draw image of confusion matrix
        color_map = LinearSegmentedColormap.from_list("white_to_blue", [(1, 1, 1), (0.2, 0.2, 1)], 256)
        n = cm.shape[0]
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, color_map)

        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(self.label_names)
        ax.set_yticklabels(self.label_names)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=40, ha="right", rotation_mode="anchor")

        min_val = np.amin(cm)
        max_val = np.amax(cm)
        break_point = (max_val + min_val) / 2

        plt.ylabel("true label")
        plt.xlabel("predicted label")

        # Loop over data dimensions and create text annotations.
        for i in range(n):
            for j in range(n):
                col = (1, 1, 1)
                if cm[i, j] <= break_point:
                    col = (0, 0, 0)
                ax.text(j, i, cm[i, j], ha="center", va="center", color=col, fontsize=12)

        fig.colorbar(im, ax=ax)

        ax.set_title("confusion matrix")
        fig.tight_layout()
        plt.savefig(self.path_to_confusion_matrix_img, bbox_inches='tight')

    def analyze(self):
        train_data, test_data, train_labels, test_labels = train_test_split(self.data, self.labels, test_size=.2,
                                                                            random_state=42)

        predicted_test_labels = self.fitted_clf.predict(test_data)

        scores = {
            "train_data_mean_accuracy": self.fitted_clf.score(train_data, train_labels),
            "test_data_mean_accuracy": np.mean(predicted_test_labels == test_labels)
        }
        scores = {k: round(v, 3) for k, v in scores.items()}

        cm = confusion_matrix(test_labels, predicted_test_labels)
        self.__build_cm_image(cm)

        # Save output files
        with open(self.path_to_best_clf_params, "w") as fp:
            json.dump(self.fitted_clf.best_params_, fp)
        with open(self.path_to_scores, "w") as fp:
            json.dump(scores, fp)


if __name__ == "__main__":
    ClassificationAnalyzer(sys.argv[1:]).analyze()

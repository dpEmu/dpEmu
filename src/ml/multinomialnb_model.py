import json
import pickle
import sys

import numpy as np
from joblib import dump
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB


class CustomMultinomialNB:

    def __init__(self, paths):
        np.random.seed(42)
        with open(paths[0], "rb") as file:
            self.data = pickle.load(file)
        self.labels = np.load(paths[1])
        with open(paths[2], "r") as file:
            self.clf_param_grid = json.load(file)
        self.path_to_vectorized_data = paths[3]
        self.path_to_fitted_clf = paths[4]

    def fit_and_optimize(self):
        vectorizer = TfidfVectorizer(max_df=0.5, min_df=2)
        vectorized_data = vectorizer.fit_transform(self.data)
        print(vectorized_data.shape)

        vectorized_train_data, _, train_labels, _ = train_test_split(vectorized_data, self.labels, test_size=.2,
                                                                     random_state=42)

        clf = MultinomialNB()
        # clf = LinearSVC(random_state=42)

        grid_search_cv = GridSearchCV(clf, self.clf_param_grid, cv=3, n_jobs=-1)
        grid_search_cv.fit(vectorized_train_data, train_labels)

        save_npz(self.path_to_vectorized_data, vectorized_data)
        dump(grid_search_cv, self.path_to_fitted_clf)


def main(argv):
    model = CustomMultinomialNB(argv[1:])
    model.fit_and_optimize()


if __name__ == "__main__":
    main(sys.argv)

import json
import pickle
import sys

import numpy as np
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


class CustomMultinomialNB:

    def __init__(self, paths):
        np.random.seed(42)
        with open(paths[0], "rb") as file:
            self.data = pickle.load(file)
        self.labels = np.load(paths[1])
        with open(paths[2], "r") as file:
            self.clf_param_grid = json.load(file)
        self.path_to_fitted_clf = paths[3]

    def fit_and_optimize(self):
        train_data, _, train_labels, _ = train_test_split(self.data, self.labels, test_size=.2, random_state=42)

        pipeline = Pipeline([
            ("tfidf_vectorizer", TfidfVectorizer(max_df=0.5, min_df=2)),
            ("multinomial_nb", MultinomialNB()),
            # ("truncated_svd", TruncatedSVD(random_state=42)),
            # ("normalizer", Normalizer(copy=False)),
            # ("linear_svc", LinearSVC(random_state=42)),
        ])

        grid_search_cv = GridSearchCV(pipeline, self.clf_param_grid, cv=3, n_jobs=-1)
        grid_search_cv.fit(train_data, train_labels)

        dump(grid_search_cv, self.path_to_fitted_clf)


def main(argv):
    model = CustomMultinomialNB(argv[1:])
    model.fit_and_optimize()


if __name__ == "__main__":
    main(sys.argv)

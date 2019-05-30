import json
import sys

import numpy as np
from sklearn.externals.joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


class CustomMultinomialNB:

    def __init__(self, paths):
        self.data = np.load(paths[0])
        self.labels = np.load(paths[1])
        with open(paths[2], "r") as file:
            self.params = json.load(file)
        self.path_to_fitted_model = paths[3]
        np.random.seed(42)

    def fit_data(self):
        train_data, test_data, train_labels, test_labels = train_test_split(self.data, self.labels, test_size=.2,
                                                                            random_state=42)

        pipeline = Pipeline([
            ("tfidf_vect", TfidfVectorizer()),
            ("naive_bayes", MultinomialNB()),
        ])

        grid_search_cv = GridSearchCV(pipeline, self.params)
        grid_search_cv.fit(train_data, train_labels)

        dump(grid_search_cv, self.path_to_fitted_model)


def main(argv):
    naive_bayes = CustomMultinomialNB([argv[1], argv[2], argv[3], argv[4]])
    naive_bayes.fit_data()


if __name__ == "__main__":
    main(sys.argv)

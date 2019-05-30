import json

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.ml.utils import run_ml_script
from src.utils import load_20newsgroups_as_npy, generate_unique_path


def main():
    path_to_data, path_to_labels = load_20newsgroups_as_npy(.05)
    path_to_params = generate_unique_path("tmp", "json")
    path_to_fitted_model = generate_unique_path("tmp", "joblib")
    path_to_scores = generate_unique_path("tmp", "json")
    path_to_best_params = generate_unique_path("tmp", "json")

    params = {
        "naive_bayes__alpha": [k / 500 for k in range(1, 500)]
    }
    with open(path_to_params, "w") as file:
        json.dump(params, file)

    run_ml_script("python src/ml/naive_bayes_model.py {} {} {} {}".format(
        path_to_data,
        path_to_labels,
        path_to_params,
        path_to_fitted_model
    ))
    run_ml_script("python src/ml/classifying_analyzer.py {} {} {} {} {}".format(
        path_to_data,
        path_to_labels,
        path_to_fitted_model,
        path_to_scores,
        path_to_best_params
    ))

    with open(path_to_scores, "r") as file:
        scores = json.load(file)
    with open(path_to_best_params, "r") as file:
        best_params = json.load(file)

    print(scores)
    print(best_params)

    # np.random.seed(42)
    # newsgroups = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"), random_state=42)
    #
    # data, _, labels, _ = train_test_split(newsgroups["data"], newsgroups["target"], train_size=.05, random_state=42)
    # train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=.2, random_state=42)
    #
    # pipeline = Pipeline([
    #     ("tfidf_vect", TfidfVectorizer()),
    #     ("naive_bayes", MultinomialNB()),
    # ])
    #
    # params = {
    #     "naive_bayes__alpha": [k / 500 for k in range(1, 500)],
    # }
    # clf = GridSearchCV(pipeline, params)
    # clf.fit(train_data, train_labels)
    #
    # print("{:.3f}".format(clf.score(test_data, test_labels)))
    # print(clf.best_params_)
    # print(clf.best_score_)


if __name__ == "__main__":
    main()

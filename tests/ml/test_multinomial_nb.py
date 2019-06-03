import json

import numpy as np

from src.ml.utils import run_ml_script
from src.utils import generate_unique_path, load_newsgroups_as_pickle


def test_naive_bayes_with_analysis():
    categories = [
        "alt.atheism",
        "talk.religion.misc",
        "comp.graphics",
        "sci.space",
    ]
    path_to_data, path_to_labels, _ = load_newsgroups_as_pickle(categories)
    path_to_clf_param_grid = generate_unique_path("tmp", "json")
    path_to_fitted_clf = generate_unique_path("tmp", "joblib")
    path_to_best_clf_params = generate_unique_path("tmp", "json")
    path_to_scores = generate_unique_path("tmp", "json")
    path_to_confusion_matrix = generate_unique_path("tmp", "npy")

    clf_param_grid = {
        "multinomial_nb__alpha": [10 ** i for i in range(-3, 1)],
    }
    with open(path_to_clf_param_grid, "w") as file:
        json.dump(clf_param_grid, file)

    run_ml_script("python src/ml/multinomial_nb_model.py {} {} {} {}".format(
        path_to_data,
        path_to_labels,
        path_to_clf_param_grid,
        path_to_fitted_clf
    ))
    run_ml_script("python src/ml/classification_analyzer.py {} {} {} {} {} {}".format(
        path_to_data,
        path_to_labels,
        path_to_fitted_clf,
        path_to_best_clf_params,
        path_to_scores,
        path_to_confusion_matrix
    ))

    with open(path_to_best_clf_params, "r") as file:
        best_clf_params = json.load(file)
    with open(path_to_scores, "r") as file:
        scores = json.load(file)
    cm = np.load(path_to_confusion_matrix)

    assert best_clf_params == {"multinomial_nb__alpha": 0.01}
    assert scores == {"train_data_mean_accuracy": 0.97, "test_data_mean_accuracy": 0.823}

    predicted_cm = np.array([
        [112, 2, 12, 29],
        [1, 182, 11, 1],
        [9, 11, 177, 4],
        [27, 4, 9, 87]
    ])

    assert np.array_equal(cm, predicted_cm)

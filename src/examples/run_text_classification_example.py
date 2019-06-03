import json

from PIL import Image

from src.ml.utils import run_ml_script
from src.utils import load_newsgroups_as_pickle, generate_unique_path


def main():
    categories = [
        "alt.atheism",
        "talk.religion.misc",
        "comp.graphics",
        "sci.space",
    ]
    path_to_data, path_to_labels, path_to_label_names = load_newsgroups_as_pickle(categories)
    # path_to_data, path_to_labels, path_to_label_names = load_newsgroups_as_pickle()
    path_to_clf_param_grid = generate_unique_path("tmp", "json")
    path_to_fitted_clf = generate_unique_path("tmp", "joblib")
    path_to_best_clf_params = generate_unique_path("tmp", "json")
    path_to_scores = generate_unique_path("tmp", "json")
    path_to_confusion_matrix_img = generate_unique_path("tmp", "png")

    clf_param_grid = {
        "multinomial_nb__alpha": [10 ** i for i in range(-3, 1)],
        # "truncated_svd__n_components": [100, 150, 200, 250, 300],
        # "linear_svc__C": [10 ** k for k in range(-3, 4)],
    }
    with open(path_to_clf_param_grid, "w") as file:
        json.dump(clf_param_grid, file)

    run_ml_script("python src/ml/multinomial_nb_model.py {} {} {} {}".format(
        path_to_data,
        path_to_labels,
        path_to_clf_param_grid,
        path_to_fitted_clf
    ))
    run_ml_script("python src/ml/classification_analyzer.py {} {} {} {} {} {} {}".format(
        path_to_data,
        path_to_labels,
        path_to_label_names,
        path_to_fitted_clf,
        path_to_best_clf_params,
        path_to_scores,
        path_to_confusion_matrix_img
    ))

    with open(path_to_best_clf_params, "r") as file:
        best_clf_params = json.load(file)
    with open(path_to_scores, "r") as file:
        scores = json.load(file)
    cm_img = Image.open(path_to_confusion_matrix_img)

    print(best_clf_params)
    print(scores)
    cm_img.show()


if __name__ == "__main__":
    main()

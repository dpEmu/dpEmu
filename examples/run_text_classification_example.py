import sys
import warnings
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from numba import NumbaDeprecationWarning, NumbaWarning
from numpy.random import RandomState
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from dpemu import runner
from dpemu import dataset_utils
from dpemu import ml_utils
from dpemu import plotting_utils
from dpemu import array
from dpemu import filters
from dpemu import radius_generators

warnings.simplefilter("ignore", category=ConvergenceWarning)
warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaWarning)


class Preprocessor:
    def __init__(self):
        self.random_state = RandomState(42)

    def run(self, train_data, test_data, _):
        vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words="english")
        vectorized_train_data = vectorizer.fit_transform(train_data)
        vectorized_test_data = vectorizer.transform(test_data)

        reduced_test_data = ml_utils.reduce_dimensions_sparse(vectorized_test_data, self.random_state)

        return vectorized_train_data, vectorized_test_data, {"reduced_test_data": reduced_test_data}


class AbstractModel(ABC):

    def __init__(self):
        self.random_state = RandomState(42)

    @abstractmethod
    def get_fitted_model(self, train_data, train_labels, params):
        pass

    def run(self, train_data, test_data, params):
        train_labels = params["train_labels"]
        test_labels = params["test_labels"]

        fitted_model = self.get_fitted_model(train_data, train_labels, params)

        predicted_test_labels = fitted_model.predict(test_data)
        cm = confusion_matrix(test_labels, predicted_test_labels)

        return {
            "confusion_matrix": cm,
            "predicted_test_labels": predicted_test_labels,
            "test_mean_accuracy": round(np.mean(predicted_test_labels == test_labels), 3),
            "train_mean_accuracy": fitted_model.score(train_data, train_labels),
        }


class MultinomialNBModel(AbstractModel):

    def __init__(self):
        super().__init__()

    def get_fitted_model(self, train_data, train_labels, params):
        return MultinomialNB(params["alpha"]).fit(train_data, train_labels)


class LinearSVCModel(AbstractModel):

    def __init__(self):
        super().__init__()

    def get_fitted_model(self, train_data, train_labels, params):
        return LinearSVC(C=params["C"], random_state=self.random_state).fit(train_data, train_labels)


def visualize(df, dataset_name, label_names, test_data, use_interactive_mode):
    plotting_utils.visualize_scores(df, ["test_mean_accuracy", "train_mean_accuracy"], [True, True], "p",
                                    f"{dataset_name} classification scores with added error")
    plotting_utils.visualize_best_model_params(df, "MultinomialNB #1", ["alpha"], ["test_mean_accuracy"], [True], "p",
                                               f"Best parameters for {dataset_name} clustering", x_log=False,
                                               y_log=True)
    plotting_utils.visualize_best_model_params(df, "MultinomialNBClean #1", ["alpha"], ["test_mean_accuracy"], [True],
                                               "p", f"Best parameters for {dataset_name} clustering", x_log=False,
                                               y_log=True)
    plotting_utils.visualize_best_model_params(df, "LinearSVC #1", ["C"], ["test_mean_accuracy"], [True], "p",
                                               f"Best parameters for {dataset_name} clustering", x_log=False,
                                               y_log=True)
    plotting_utils.visualize_best_model_params(df, "LinearSVCClean #1", ["C"], ["test_mean_accuracy"], [True], "p",
                                               f"Best parameters for {dataset_name} clustering", x_log=False,
                                               y_log=True)
    plotting_utils.visualize_classes(df, label_names, "p", "reduced_test_data", "test_labels", "tab20",
                                     f"{dataset_name} (n={len(test_data)}) classes with added error")

    if use_interactive_mode:
        def on_click(element, label, predicted_label):
            print(label, " predicted as ", predicted_label, ":", sep="")
            print(element, end="\n\n")
    else:
        on_click = None

    plotting_utils.visualize_confusion_matrices(df, label_names, "test_mean_accuracy", True, "p", "test_labels",
                                                "predicted_test_labels", on_click)

    plt.show()


def main(argv):
    if len(argv) not in [3, 4] or argv[1] not in ["all", "test"]:
        exit(0)
    if len(argv) == 4 and argv[3] == "-i":
        use_interactive_mode = True
    else:
        use_interactive_mode = False

    data, labels, label_names, dataset_name = dataset_utils.load_newsgroups(argv[1], int(argv[2]))
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=.2,
                                                                        random_state=RandomState(42))

    err_root_node = array.Array()
    err_root_node.addfilter(filters.MissingArea("p", "radius_generator", "missing_value"))
    # err_root_node.addfilter(OCRError("normalized_params", "p"))

    p_steps = np.linspace(0, .28, num=8)
    err_params_list = [{
        "p": p,
        "radius_generator": radius_generators.GaussianRadiusGenerator(0, 1),
        "missing_value": " "
    } for p in p_steps]

    # p_steps = np.linspace(0, 1, num=11)
    # params = load_ocr_error_params("config/example_text_error_params.json")
    # normalized_params = normalize_ocr_error_params(params)
    # err_params_list = [{
    #     "p": p,
    #     "normalized_params": normalized_params
    # } for p in p_steps]

    alpha_steps = [10 ** i for i in range(-2, 1)]
    C_steps = [10 ** k for k in range(-2, 1)]
    model_params_base = {"train_labels": train_labels, "test_labels": test_labels}
    model_params_dict_list = [
        {
            "model": MultinomialNBModel,
            "params_list": [{"alpha": alpha, **model_params_base} for alpha in alpha_steps],
            "use_clean_train_data": False
        },
        {
            "model": MultinomialNBModel,
            "params_list": [{"alpha": alpha, **model_params_base} for alpha in alpha_steps],
            "use_clean_train_data": True
        },
        {
            "model": LinearSVCModel,
            "params_list": [{"C": C, **model_params_base} for C in C_steps],
            "use_clean_train_data": False
        },
        {
            "model": LinearSVCModel,
            "params_list": [{"C": C, **model_params_base} for C in C_steps],
            "use_clean_train_data": True
        },
    ]

    df = runner.run(train_data, test_data, Preprocessor, None, err_root_node, err_params_list, model_params_dict_list,
                    use_interactive_mode=use_interactive_mode)

    plotting_utils.print_results(df, ["train_labels", "test_labels", "reduced_test_data", "confusion_matrix",
                                      "predicted_test_labels", "radius_generator", "missing_value",
                                      "normalized_params"])

    visualize(df, dataset_name, label_names, test_data, use_interactive_mode)


if __name__ == "__main__":
    main(sys.argv)

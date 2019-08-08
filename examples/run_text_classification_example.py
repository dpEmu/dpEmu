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

from dpemu import runner_
from dpemu.datasets.utils import load_newsgroups
from dpemu.ml.utils import reduce_dimensions_sparse
from dpemu.plotting.utils import visualize_best_model_params
from dpemu.plotting.utils import visualize_scores, visualize_classes, print_results_by_model, \
    visualize_confusion_matrices
from dpemu.problemgenerator.array import Array
from dpemu.problemgenerator.filters import MissingArea
from dpemu.problemgenerator.radius_generators import GaussianRadiusGenerator

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

        reduced_test_data = reduce_dimensions_sparse(vectorized_test_data, self.random_state)

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


def get_data(argv):
    data, labels, label_names, dataset_name = load_newsgroups(argv[1], int(argv[2]))
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=.2,
                                                                        random_state=RandomState(42))
    return train_data, test_data, train_labels, test_labels, label_names, dataset_name


def get_err_root_node():
    err_root_node = Array()
    err_root_node.addfilter(MissingArea("p", "radius_generator", "missing_value"))
    # err_root_node.addfilter(OCRError("normalized_params", "p"))
    return err_root_node


def get_err_params_list():
    p_steps = np.linspace(0, .28, num=8)
    err_params_list = [{
        "p": p,
        "radius_generator": GaussianRadiusGenerator(0, 1),
        "missing_value": " "
    } for p in p_steps]

    # p_steps = np.linspace(0, 1, num=11)
    # params = load_ocr_error_params("config/example_text_error_params.json")
    # normalized_params = normalize_ocr_error_params(params)
    # err_params_list = [{
    #     "p": p,
    #     "normalized_params": normalized_params
    # } for p in p_steps]

    return err_params_list


def get_model_params_dict_list(train_labels, test_labels):
    alpha_steps = [10 ** i for i in range(-2, 1)]
    C_steps = [10 ** k for k in range(-2, 1)]
    model_params_base = {"train_labels": train_labels, "test_labels": test_labels}
    return [
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


def visualize(df, dataset_name, label_names, test_data, use_interactive_mode):
    visualize_scores(df, ["test_mean_accuracy", "train_mean_accuracy"], [True, True], "p",
                     f"{dataset_name} classification scores with added error")
    visualize_best_model_params(df, "MultinomialNB #1", ["alpha"], ["test_mean_accuracy"], [True], "p",
                                f"Best parameters for {dataset_name} clustering", x_log=False, y_log=True)
    visualize_best_model_params(df, "MultinomialNBClean #1", ["alpha"], ["test_mean_accuracy"], [True], "p",
                                f"Best parameters for {dataset_name} clustering", x_log=False, y_log=True)
    visualize_best_model_params(df, "LinearSVC #1", ["C"], ["test_mean_accuracy"], [True], "p",
                                f"Best parameters for {dataset_name} clustering", x_log=False, y_log=True)
    visualize_best_model_params(df, "LinearSVCClean #1", ["C"], ["test_mean_accuracy"], [True], "p",
                                f"Best parameters for {dataset_name} clustering", x_log=False, y_log=True)
    visualize_classes(df, label_names, "p", "reduced_test_data", "test_labels", "tab20",
                      f"{dataset_name} (n={len(test_data)}) classes with added error")

    if use_interactive_mode:
        def on_click(element, label, predicted_label):
            print(label, " predicted as ", predicted_label, ":", sep="")
            print(element, end="\n\n")
    else:
        on_click = None

    visualize_confusion_matrices(df, label_names, "test_mean_accuracy", True, "p", "test_labels",
                                 "predicted_test_labels", on_click)
    plt.show()


def main(argv):
    if len(argv) not in [3, 4] or argv[1] not in ["all", "test"]:
        exit(0)
    if len(argv) == 4 and argv[3] == "-i":
        use_interactive_mode = True
    else:
        use_interactive_mode = False

    train_data, test_data, train_labels, test_labels, label_names, dataset_name = get_data(argv)

    df = runner_.run(
        train_data=train_data,
        test_data=test_data,
        preproc=Preprocessor,
        preproc_params=None,
        err_root_node=get_err_root_node(),
        err_params_list=get_err_params_list(),
        model_params_dict_list=get_model_params_dict_list(train_labels, test_labels),
        use_interactive_mode=use_interactive_mode
    )

    print_results_by_model(df, ["train_labels", "test_labels", "reduced_test_data", "confusion_matrix",
                                "predicted_test_labels", "radius_generator", "missing_value", "normalized_params"])
    visualize(df, dataset_name, label_names, test_data, use_interactive_mode)


if __name__ == "__main__":
    main(sys.argv)

import sys
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src import runner_
from src.datasets.utils import load_newsgroups
from src.plotting.utils import visualize_scores, print_dfs, visualize_confusion_matrices
from src.problemgenerator import array, copy, filters
from src.problemgenerator.radius_generators import GaussianRadiusGenerator
from src.utils import split_df_by_model


class AbstractModel(ABC):

    def __init__(self):
        self.random_state = np.random.RandomState(42)

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
            "test_mean_accuracy": round(np.mean(predicted_test_labels == test_labels), 3),
            "train_mean_accuracy": fitted_model.score(train_data, train_labels),
            "confusion_matrix": cm,
        }


class MultinomialNBModel(AbstractModel):

    def __init__(self):
        super().__init__()

    def get_fitted_model(self, train_data, train_labels, params):
        return Pipeline([
            ("tfidf_vectorizer", TfidfVectorizer(max_df=0.5, min_df=2)),
            ("multinomial_nb", MultinomialNB(params["alpha"])),
        ]).fit(train_data, train_labels)


class LinearSVCModel(AbstractModel):

    def __init__(self):
        super().__init__()

    def get_fitted_model(self, train_data, train_labels, params):
        return Pipeline([
            ("tfidf_vectorizer", TfidfVectorizer(max_df=0.5, min_df=2)),
            ("linear_svc", LinearSVC(C=params["C"], random_state=self.random_state)),
        ]).fit(train_data, train_labels)


class ErrGen:
    def __init__(self):
        self.random_state = np.random.RandomState(42)

    def generate_error(self, data, params):
        data = np.array(data)

        data_node = array.Array(data.shape)
        root_node = copy.Copy(data_node)

        f = filters.MissingArea(params["p"], params["radius_generator"], params["missing_value"])
        data_node.addfilter(f)

        return root_node.process(data, self.random_state)


def visualize(df, dataset_name, label_names):
    dfs = split_df_by_model(df)

    print_dfs(dfs, ["train_labels", "test_labels", "confusion_matrix", "radius_generator", "missing_value"])

    visualize_scores(
        dfs,
        ["test_mean_accuracy", "train_mean_accuracy"],
        "p",
        f"{dataset_name} classification scores with added missing areas"
    )
    visualize_confusion_matrices(
        dfs,
        label_names,
        "test_mean_accuracy",
        "p",
    )
    plt.show()


def main(argv):
    if len(argv) == 1:
        data, labels, label_names, dataset_name = load_newsgroups()
    if len(argv) == 2:
        data, labels, label_names, dataset_name = load_newsgroups(int(argv[1]))
    else:
        exit(0)

    train_data, test_data, train_labels, test_labels = train_test_split(
        data,
        labels,
        test_size=.2,
        random_state=np.random.RandomState(42)
    )

    p_steps = np.linspace(0, .4, num=6)
    err_params_list = [{
        "p": p,
        "radius_generator": GaussianRadiusGenerator(0, 1),
        "missing_value": " "
    } for p in p_steps]

    alpha_steps = [10 ** i for i in range(-3, 1)]
    # C_steps = [10 ** k for k in range(-3, 4)]
    model_params_base = {
        "train_labels": train_labels,
        "test_labels": test_labels,
    }
    model_params_tuple_list = [
        (MultinomialNBModel, [{"alpha": alpha, **model_params_base} for alpha in alpha_steps]),
        (MultinomialNBModel, [{"alpha": alpha, **model_params_base} for alpha in alpha_steps], True),
        # (LinearSVCModel, [{"C": C, **model_params_base} for C in C_steps]),
        # (LinearSVCModel, [{"C": C, **model_params_base} for C in C_steps], True),
    ]

    df = runner_.run(train_data, test_data, ErrGen, err_params_list, model_params_tuple_list)

    visualize(df, dataset_name, label_names)


if __name__ == "__main__":
    main(sys.argv)

from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from joblib import dump
from matplotlib.colors import LinearSegmentedColormap
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src import runner_
from src.plotting.utils import visualize_scores
from src.problemgenerator import array, copy, filters
from src.problemgenerator.radius_generators import GaussianRadiusGenerator
from src.utils import split_df_by_model, generate_unique_path


class AbstractModel(ABC):

    def __init__(self):
        self.random_state = np.random.RandomState(42)

    @abstractmethod
    def get_fitted_model(self, train_data, train_labels, model_params):
        pass

    def run(self, data, model_params):
        labels = model_params["labels"]
        train_data, test_data, train_labels, test_labels = train_test_split(
            data,
            labels,
            test_size=.2,
            random_state=self.random_state
        )

        fitted_model = self.get_fitted_model(train_data, train_labels, model_params)
        path_to_fitted_model = generate_unique_path("models", "joblib")
        dump(fitted_model, path_to_fitted_model)

        predicted_test_labels = fitted_model.predict(test_data)
        cm = confusion_matrix(test_labels, predicted_test_labels)

        return {
            "test_data_mean_accuracy": round(np.mean(predicted_test_labels == test_labels), 3),
            "train_data_mean_accuracy": fitted_model.score(train_data, train_labels),
            "confusion_matrix": cm,
        }


class MultinomialNBModel(AbstractModel):

    def __init__(self):
        super().__init__()

    def get_fitted_model(self, train_data, train_labels, model_params):
        return Pipeline([
            ("tfidf_vectorizer", TfidfVectorizer(max_df=0.5, min_df=2)),
            ("multinomial_nb", MultinomialNB(model_params["alpha"])),
        ]).fit(train_data, train_labels)


class MultinomialNBCleanModel(AbstractModel):

    def __init__(self):
        super().__init__()

    def get_fitted_model(self, train_data, train_labels, model_params):
        print(train_data[0])
        train_data, _, train_labels, _ = train_test_split(
            model_params["data"],
            model_params["labels"],
            test_size=.2,
            random_state=self.random_state
        )
        print(train_data[0])
        return Pipeline([
            ("tfidf_vectorizer", TfidfVectorizer(max_df=0.5, min_df=2)),
            ("multinomial_nb", MultinomialNB(model_params["alpha"])),
        ]).fit(train_data, train_labels)


class LinearSVCModel(AbstractModel):

    def __init__(self):
        super().__init__()

    def get_fitted_model(self, train_data, train_labels, model_params):
        return Pipeline([
            ("tfidf_vectorizer", TfidfVectorizer(max_df=0.5, min_df=2)),
            ("linear_svc", LinearSVC(C=model_params["C"], random_state=self.random_state)),
        ]).fit(train_data, train_labels)


class LinearSVCCleanModel(AbstractModel):

    def __init__(self):
        super().__init__()

    def get_fitted_model(self, train_data, train_labels, model_params):
        train_data, _, train_labels, _ = train_test_split(
            model_params["data"],
            model_params["labels"],
            test_size=.2,
            random_state=self.random_state
        )
        return Pipeline([
            ("tfidf_vectorizer", TfidfVectorizer(max_df=0.5, min_df=2)),
            ("linear_svc", LinearSVC(C=model_params["C"], random_state=self.random_state)),
        ]).fit(train_data, train_labels)


def load_newsgroups(categories=None):
    newsgroups = fetch_20newsgroups(subset="test", categories=categories, remove=("headers", "footers", "quotes"),
                                    random_state=np.random.RandomState(42))
    return newsgroups["data"], np.array(newsgroups["target"].astype(int)), newsgroups["target_names"], "20newsgroups"


class ErrGen:
    def __init__(self, data):
        self.data = data

    def generate_error(self, params):
        data = np.array(self.data)

        data_node = array.Array(data.shape)
        root_node = copy.Copy(data_node)

        f = filters.MissingArea(params["p"], params["radius_generator"], params["missing_value"])
        data_node.addfilter(f)

        # return root_node.process(data, np.random.RandomState(42))
        err_data = root_node.process(data, np.random.RandomState(42))
        return err_data


def visualize_confusion_matrix(cm, label_names, title):
    # Draw image of confusion matrix
    color_map = LinearSegmentedColormap.from_list("white_to_blue", [(1, 1, 1), (0.2, 0.2, 1)], 256)
    n = cm.shape[0]
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, color_map)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(label_names)
    ax.set_yticklabels(label_names)

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

    ax.set_title(title)
    fig.tight_layout()
    path_to_plot = generate_unique_path("out", "png")
    plt.savefig(path_to_plot, bbox_inches="tight")


def visualize_confusion_matrices(dfs, label_names, score_name, err_param_name):
    for df in dfs:
        df_ = df.loc[df.groupby(err_param_name, sort=False)[score_name].idxmax()].reset_index(drop=True)
        for i in range(df_.shape[0]):
            visualize_confusion_matrix(
                df_["confusion_matrix"][i],
                label_names,
                f"{df.name} confusion matrix ({err_param_name}={df_[err_param_name][i]})",
            )


def visualize(dfs, dataset_name, label_names):
    visualize_scores(
        dfs,
        ["test_data_mean_accuracy", "train_data_mean_accuracy"],
        "p",
        f"{dataset_name} classification scores with added missing areas"
    )
    # visualize_confusion_matrices(
    #     dfs,
    #     label_names,
    #     "test_data_mean_accuracy",
    #     "p",
    # )
    plt.show()


def main():
    categories = [
        "alt.atheism",
        "talk.religion.misc",
        "comp.graphics",
        "sci.space",
    ]
    data, labels, label_names, dataset_name = load_newsgroups(categories)

    # p_steps = np.linspace(0, .3, num=5)
    p_steps = np.linspace(0, .3, num=1)
    err_params_list = [{
        "p": p,
        "radius_generator": GaussianRadiusGenerator(0, 1),
        "missing_value": " "
    } for p in p_steps]

    alpha_steps = [10 ** i for i in range(-3, 1)]
    C_steps = [10 ** k for k in range(-3, 4)]
    model_param_pairs = [
        # (MultinomialNBModel, [{"alpha": alpha, "labels": labels} for alpha in alpha_steps]),
        (MultinomialNBCleanModel, [{"alpha": alpha, "data": data, "labels": labels} for alpha in alpha_steps]),
        # (LinearSVCModel, [{"C": C, "labels": labels} for C in C_steps]),
        # (LinearSVCCleanModel, [{"C": C, "data": data, "labels": labels} for C in C_steps]),
    ]

    df = runner_.run(ErrGen(data), err_params_list, model_param_pairs)
    dfs = split_df_by_model(df)

    for df in dfs:
        print(df.name)
        print(df.drop(columns=["labels", "confusion_matrix", "radius_generator"]))

    visualize(dfs, dataset_name, label_names)


if __name__ == "__main__":
    main()

# MIT License
#
# Copyright (c) 2019 Tuomas Halvari, Juha Harviainen, Juha Mylläri, Antti Röyskö, Juuso Silvennoinen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys
import warnings
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from hdbscan import HDBSCAN
from numba.errors import NumbaDeprecationWarning, NumbaWarning
from numpy.random import RandomState
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

from dpemu import runner
from dpemu.dataset_utils import load_digits_, load_mnist_unsplit, load_fashion
from dpemu.filters.image import Rotation
from dpemu.ml_utils import reduce_dimensions
from dpemu.nodes import Array
from dpemu.nodes.series import Series
from dpemu.plotting_utils import visualize_best_model_params, visualize_scores, visualize_classes, \
    print_results_by_model, visualize_interactive_plot

warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaWarning)


def get_data(argv):
    if argv[1] == "digits":
        data, labels, label_names, dataset_name = load_digits_(int(argv[2]))
    elif argv[1] == "mnist":
        data, labels, label_names, dataset_name = load_mnist_unsplit(int(argv[2]))
    else:
        data, labels, label_names, dataset_name = load_fashion(int(argv[2]))
    return data, labels, label_names, dataset_name


def get_err_root_node():
    err_img_node = Array(reshape=(28, 28))
    err_root_node = Series(err_img_node)
    # err_img_node.addfilter(GaussianNoise("mean", "std"))
    # err_img_node.addfilter(Clip("min_val", "max_val"))
    # err_img_node.addfilter(Missing("probability", "missing_value_id"))
    err_img_node.addfilter(Rotation("min_angle", "max_angle"))
    return err_root_node


def get_err_params_list(data):
    # min_val = np.amin(data)
    # max_val = np.amax(data)
    # std_steps = np.linspace(0, max_val, num=8)
    # err_params_list = [{"mean": 0, "std": std, "min_val": min_val, "max_val": max_val} for std in std_steps]
    # p_steps = np.linspace(0, .5, num=6)
    # err_params_list = [{"probability": p, "missing_value_id": 0} for p in p_steps]
    angle_steps = np.linspace(0, 84, num=8)
    err_params_list = [{"min_angle": -a, "max_angle": a} for a in angle_steps]
    return err_params_list


class Preprocessor:
    def __init__(self):
        self.random_state = RandomState(42)

    def run(self, _, data, params):
        reduced_data = reduce_dimensions(data, self.random_state)
        return None, reduced_data, {"reduced_data": reduced_data}


class AbstractModel(ABC):

    def __init__(self):
        self.random_state = RandomState(42)

    @abstractmethod
    def get_fitted_model(self, data, params):
        pass

    def run(self, _, data, params):
        labels = params["labels"]
        fitted_model = self.get_fitted_model(data, params)
        return {
            "AMI": round(adjusted_mutual_info_score(labels, fitted_model.labels_, average_method="arithmetic"), 3),
            "ARI": round(adjusted_rand_score(labels, fitted_model.labels_), 3),
        }


class KMeansModel(AbstractModel):

    def __init__(self):
        super().__init__()

    def get_fitted_model(self, data, params):
        labels = params["labels"]
        n_classes = len(np.unique(labels))
        return KMeans(n_clusters=n_classes, random_state=self.random_state).fit(data)


class AgglomerativeModel(AbstractModel):

    def __init__(self):
        super().__init__()

    def get_fitted_model(self, data, params):
        labels = params["labels"]
        n_classes = len(np.unique(labels))
        return AgglomerativeClustering(n_clusters=n_classes).fit(data)


class HDBSCANModel(AbstractModel):

    def __init__(self):
        super().__init__()

    def get_fitted_model(self, data, params):
        return HDBSCAN(
            min_samples=params["min_samples"],
            min_cluster_size=params["min_cluster_size"],
            core_dist_n_jobs=1
        ).fit(data)


def get_model_params_dict_list(data, labels):
    n_data = data.shape[0]
    divs = [12, 25, 50]
    min_cluster_size_steps = [round(n_data / div) for div in divs]
    min_samples_steps = [1, 10]
    return [
        {"model": KMeansModel, "params_list": [{"labels": labels}]},
        {"model": AgglomerativeModel, "params_list": [{"labels": labels}]},
        {"model": HDBSCANModel, "params_list": [{
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "labels": labels
        } for min_cluster_size in min_cluster_size_steps for min_samples in min_samples_steps]},
    ]


def visualize(df, label_names, dataset_name, data, use_interactive_mode):
    visualize_scores(
        df,
        score_names=["AMI", "ARI"],
        is_higher_score_better=[True, True],
        # err_param_name="std",
        # err_param_name="probability",
        err_param_name="max_angle",
        # title=f"{dataset_name} clustering scores with added gaussian noise",
        # title=f"{dataset_name} clustering scores with missing pixels",
        title=f"{dataset_name} clustering scores with rotation",
    )
    visualize_best_model_params(
        df,
        model_name="HDBSCAN",
        model_params=["min_cluster_size", "min_samples"],
        score_names=["AMI", "ARI"],
        is_higher_score_better=[True, True],
        # err_param_name="std",
        # err_param_name="probability",
        err_param_name="max_angle",
        title=f"Best parameters for {dataset_name} clustering"
    )
    visualize_classes(
        df,
        label_names,
        # err_param_name="std",
        # err_param_name="probability",
        err_param_name="max_angle",
        reduced_data_column="reduced_data",
        labels_column="labels",
        cmap="tab10",
        # title=f"{dataset_name} (n={data.shape[0]}) classes with added gaussian noise"
        # title=f"{dataset_name} (n={data.shape[0]}) classes with missing pixels"
        title=f"{dataset_name} (n={data.shape[0]}) classes with rotation"
    )

    if use_interactive_mode:
        def on_click(original, modified):
            # reshape data
            original = original.reshape((28, 28))
            modified = modified.reshape((28, 28))

            # create a figure and draw the images
            fg, axs = plt.subplots(1, 2)
            axs[0].matshow(original, cmap='gray_r')
            axs[0].axis('off')
            axs[1].matshow(modified, cmap='gray_r')
            axs[1].axis('off')
            fg.show()

        # Remember to enable runner's interactive mode
        visualize_interactive_plot(df, "max_angle", data, "tab10", "reduced_data", on_click)

    plt.show()


def main(argv):
    if len(argv) not in [3, 4] or argv[1] not in ["digits", "mnist", "fashion"]:
        exit(0)
    if len(argv) == 4 and argv[3] == "-i":
        use_interactive_mode = True
    else:
        use_interactive_mode = False

    data, labels, label_names, dataset_name = get_data(argv)

    df = runner.run(
        train_data=None,
        test_data=data,
        preproc=Preprocessor,
        preproc_params=None,
        err_root_node=get_err_root_node(),
        err_params_list=get_err_params_list(data),
        model_params_dict_list=get_model_params_dict_list(data, labels),
        use_interactive_mode=use_interactive_mode
    )

    print_results_by_model(df, ["missing_value_id", "min_val", "max_val", "labels", "reduced_data"])
    visualize(df, label_names, dataset_name, data, use_interactive_mode)


if __name__ == "__main__":
    main(sys.argv)

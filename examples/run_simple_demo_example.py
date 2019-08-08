import sys

import matplotlib.pyplot as plt
import numpy as np

from dpemu import runner_
from dpemu import plotting_utils
from dpemu import array
from dpemu import filters


class Preprocessor:
    def run(self, train_data, test_data, _):
        # Return the original data without preprocessing
        return train_data, test_data, {}


class PredictorModel:
    def run(self, train_data, test_data, params):
        # The model tries to predict the values of test_data
        # by using a weighted average of previous values
        estimate = 0
        squared_error = 0

        for elem in test_data:
            # Calculate error
            squared_error += (elem - estimate) * (elem - estimate)
            # Update estimate
            estimate = (1 - params["weight"]) * estimate + params["weight"] * elem

        mean_squared_error = squared_error / len(test_data)

        return {"MSE": mean_squared_error}


def main(argv):
    # Create some fake data
    if len(argv) == 2:
        train_data = None
        test_data = np.arange(int(sys.argv[1]))
    else:
        exit(0)

    # Create error generation tree that has an Array node
    # as its root node and a GaussianNoise filter
    err_root_node = array.Array()
    err_root_node.addfilter(filters.GaussianNoise("mean", "std"))

    # The standard deviation goes from 0 to 20
    err_params_list = [{"mean": 0, "std": std} for std in range(0, 21)]

    # The model is run with different weighted estimates
    model_params_dict_list = [{
        "model": PredictorModel,
        "params_list": [{'weight': w} for w in [0.0, 0.05, 0.15, 0.5, 1.0]],
        "use_clean_train_data": False
    }]

    # Run the whole thing and get DataFrame for visualization
    df = runner_.run(train_data,
                     test_data,
                     Preprocessor,
                     None,
                     err_root_node,
                     err_params_list,
                     model_params_dict_list,
                     use_interactive_mode=True)

    # Visualize mean squared error for all used standard deviations
    plotting_utils.visualize_scores(df, ["MSE"], [False], "std", "Mean squared error")
    plt.show()


if __name__ == "__main__":
    main(sys.argv)

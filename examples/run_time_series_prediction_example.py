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

import random as rn
import sys
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from dpemu import pg_utils
from dpemu import runner
from dpemu.filters.common import GaussianNoise
from dpemu.nodes import Array
from dpemu.plotting_utils import print_results_by_model, visualize_scores, visualize_time_series_prediction


def get_data(argv):
    dataset_name = argv[1]
    n_data = int(argv[2])
    if dataset_name == "passengers":
        data = pd.read_csv("data/passengers.csv", header=0, usecols=["passengers"])[:n_data].values.astype(float)
        n_period = 12
    else:
        data = pd.read_csv("data/temperature.csv", header=0, usecols=[dataset_name])[:n_data].values.astype(float)
        n_period = 24

    data = data[~np.isnan(data)]
    n_data = len(data)
    n_test = int(n_data * .2)
    return data[:-n_test], data[-n_test:], n_data, n_period, dataset_name


def get_err_root_node():
    err_root_node = Array()
    err_root_node.addfilter(GaussianNoise("mean", "std"))
    # err_root_node.addfilter(Gap("prob_break", "prob_recover", "missing_value"))
    return err_root_node


def get_err_params_list():
    err_params_list = [{"mean": 0, "std": std} for std in range(8)]
    # err_params_list = [{"prob_break": p, "prob_recover": .5, "missing_value": np.nan} for p in [0, .03, .06, .09]]
    return err_params_list


class Preprocessor:
    def run(self, train_data, test_data, params):
        return train_data, test_data, {}


class LSTMModel:

    def __init__(self):
        seed = 42
        rn.seed(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)
        conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        session = tf.Session(graph=tf.get_default_graph(), config=conf)
        backend.set_session(session)

    @staticmethod
    def __get_periodic_diffs(data, n_period):
        return np.array([data[i] - data[i - n_period] for i in range(n_period, len(data))])

    @staticmethod
    def __get_rmse(test_pred, test):
        return sqrt(mean_squared_error(test_pred, test))

    def run(self, train_data, test_data, params):
        n_period = params["n_period"]
        train_data = train_data[~np.isnan(train_data)]
        train_data = np.reshape(train_data[~np.isnan(train_data)], (len(train_data), 1))
        test_data = test_data[~np.isnan(test_data)]
        test_data = np.reshape(test_data[~np.isnan(test_data)], (len(test_data), 1))

        n_features = 1
        n_steps = 3 * n_period
        n_nodes = 100
        n_epochs = 200

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_train = scaler.fit_transform(train_data)
        train_periodic_diffs = self.__get_periodic_diffs(scaled_train, n_period)
        train_periodic_diffs = pg_utils.to_time_series_x_y(train_periodic_diffs, n_steps)

        model = Sequential()
        model.add(LSTM(n_nodes, activation="relu", input_shape=(n_steps, n_features)))
        model.add(Dense(n_nodes, activation="relu"))
        model.add(Dense(1))
        model.compile(loss="mse", optimizer="adam")
        model.fit(train_periodic_diffs[0], train_periodic_diffs[1], epochs=n_epochs)

        train_with_test_pred = scaled_train
        for _ in range(len(test_data)):
            x_cur = self.__get_periodic_diffs(train_with_test_pred, n_period)[-n_steps:]
            x_cur = np.reshape(x_cur, (1, n_steps, n_features))
            y_cur = model.predict(x_cur) + train_with_test_pred[-n_period]
            train_with_test_pred = np.concatenate([train_with_test_pred, y_cur], axis=0)
        train_with_test_pred = scaler.inverse_transform(train_with_test_pred)

        rmse = self.__get_rmse(train_with_test_pred[-len(test_data):], test_data)
        return {
            "rmse": rmse,
            "err_data": np.concatenate([train_data, test_data], axis=0),
            "train_with_test_pred": train_with_test_pred
        }


def get_model_params_dict_list(n_period):
    return [{"model": LSTMModel, "params_list": [{"n_period": n_period}]}]


def visualize(df, n_data, dataset_name):
    visualize_scores(
        df,
        score_names=["rmse"],
        is_higher_score_better=[False],
        err_param_name="std",
        # err_param_name="prob_break",
        title=f"Prediction scores for {dataset_name} dataset (n={n_data}) with added error"
    )
    visualize_time_series_prediction(
        df,
        "LSTM",
        score_name="rmse",
        is_higher_score_better=False,
        err_param_name="std",
        # err_param_name="prob_break",
        err_data_column="err_data",
        predictions_column="train_with_test_pred",
        title=f"Predictions for {dataset_name} dataset (n={n_data}) with added error"
    )
    plt.show()


def main(argv):
    if len(argv) != 3 or argv[1] not in ["passengers", "Jerusalem", "Eilat", "Miami", "Tel Aviv District"]:
        exit(0)

    train_data, test_data, n_data, n_period, dataset_name = get_data(argv)

    df = runner.run(
        train_data=train_data,
        test_data=test_data,
        preproc=Preprocessor,
        preproc_params={},
        err_root_node=get_err_root_node(),
        err_params_list=get_err_params_list(),
        model_params_dict_list=get_model_params_dict_list(n_period),
    )

    print_results_by_model(df, dropped_columns=["err_data", "train_with_test_pred"])
    visualize(df, n_data, dataset_name)


if __name__ == "__main__":
    main(sys.argv)

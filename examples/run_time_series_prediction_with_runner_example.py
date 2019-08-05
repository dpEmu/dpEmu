import random as rn
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from keras import backend
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import src.problemgenerator.array as array
import src.problemgenerator.filters as filters
from src import runner_
from src.problemgenerator.utils import to_time_series_x_y


class Preprocessor:
    def run(self, train_data, test_data):
        return train_data, test_data, {}


class Model:
    @staticmethod
    def __get_periodic_diffs(data, n_period):
        return np.array([data[i] - data[i - n_period] for i in range(n_period, len(data))])

    @staticmethod
    def __get_rmse(test_pred, test):
        return sqrt(mean_squared_error(test_pred, test))

    @staticmethod
    def __get_plot(data, train_with_test_pred, rmse):
        plt.plot(data, label="data")
        plt.plot(train_with_test_pred, label="pred", zorder=1)
        plt.legend()
        plt.title("RMSE: {}".format(rmse))
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        byte_img = buf.read()
        byte_img = BytesIO(byte_img)
        return Image.open(byte_img)

    def run(self, _, data, model_params):
        seed = model_params["seed"]
        rn.seed(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)
        conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        session = tf.Session(graph=tf.get_default_graph(), config=conf)
        backend.set_session(session)

        # plt.plot(data)
        # plt.tight_layout()
        # plt.show()

        data = data[~np.isnan(data)]
        data = np.reshape(data, (len(data), 1))

        scaler = MinMaxScaler(feature_range=(0, 1))

        n_features = 1
        n_test = int(len(data) * .33)
        n_period = 24
        # n_period = 12
        n_steps = 3 * n_period
        n_nodes = 100
        n_epochs = 200

        train, test = data[:-n_test], data[-n_test:]
        train = scaler.fit_transform(train)
        train_periodic_diffs = self.__get_periodic_diffs(train, n_period)
        train_periodic_diffs = to_time_series_x_y(train_periodic_diffs, n_steps)

        model = Sequential()
        model.add(LSTM(n_nodes, activation="relu", input_shape=(n_steps, n_features)))
        model.add(Dense(n_nodes, activation="relu"))
        model.add(Dense(1))
        model.compile(loss="mse", optimizer="adam")
        model.fit(train_periodic_diffs[0], train_periodic_diffs[1], epochs=n_epochs)

        train_with_test_pred = train
        for _ in range(n_test):
            x_cur = self.__get_periodic_diffs(train_with_test_pred, n_period)[-n_steps:]
            x_cur = np.reshape(x_cur, (1, n_steps, n_features))
            y_cur = model.predict(x_cur) + train_with_test_pred[-n_period]
            train_with_test_pred = np.concatenate([train_with_test_pred, y_cur], axis=0)
        train_with_test_pred = scaler.inverse_transform(train_with_test_pred)

        rmse = self.__get_rmse(train_with_test_pred[-n_test:], test)
        plot = self.__get_plot(data, train_with_test_pred, rmse)
        plot.show()
        return {
            "prediction_img": plot,
            "rmse": rmse
        }


# Example usage
def main():
    data = pd.read_csv("data/passengers.csv", header=0, usecols=["passengers"]).values.astype(float)
    # data = pd.read_csv("data/temperature.csv", header=0, usecols=["Jerusalem"])[:200].values.astype(float)

    model_params_list = [
        {"model": Model, "params_list": [{"seed": 42}]}
    ]

    root_node = array.Array()

    # root_node.addfilter(filters.GaussianNoise("mean", "std"))
    root_node.addfilter(filters.SensorDrift("magnitude"))
    # root_node.addfilter(filters.Gap("prob_break", "prob_recover", "value"))

    # err_params_list = [{"mean": a, "std": b} for (a, b) in [(0, 0), (0, 15), (0, 20)]]
    err_params_list = [{"magnitude": a} for a in [0, 2, 10]]
    # err_params_list = [{"prob_break": a, "prob_recover": b, "value": np.nan}
    #                    for (a, b) in [(0, 1), (.02, .8), (.1, .5)]]

    res = runner_.run(None, data, Preprocessor, root_node, err_params_list, model_params_list)
    print(res)


if __name__ == "__main__":
    # Call main
    main()

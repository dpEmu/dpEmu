import random as rn
from collections import OrderedDict
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

from src.problemgenerator.utils import to_time_series_x_y
import src.problemgenerator.array as array
import src.problemgenerator.filters as filters
import src.problemgenerator.series as series
import src.problemgenerator.copy as copy



class Model:
    def __init__(self, data):
        seed = 42
        rn.seed(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)
        conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        session = tf.Session(graph=tf.get_default_graph(), config=conf)
        backend.set_session(session)

        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data = data[~np.isnan(data)]
        self.data = np.reshape(self.data, (len(self.data), 1))

        # plt.plot(self.data)
        # plt.tight_layout()
        # plt.show()

    @staticmethod
    def __get_periodic_diffs(data, n_period):
        return np.array([data[i] - data[i - n_period] for i in range(n_period, len(data))])

    def __get_plot(self, train_with_test_pred):
        plt.plot(self.data, label="data")
        plt.plot(train_with_test_pred, label="pred", zorder=1)
        plt.legend()
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        byte_img = buf.read()
        byte_img = BytesIO(byte_img)
        return Image.open(byte_img)

    def run(self):
        n_features = 1
        n_test = int(len(self.data) * .33)
        n_period = 24
        n_steps = 3 * n_period
        n_nodes = 100
        n_epochs = 200

        train, test = self.data[:-n_test], self.data[-n_test:]
        train = self.scaler.fit_transform(train)
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
        train_with_test_pred = self.scaler.inverse_transform(train_with_test_pred)

        out = OrderedDict()
        out["prediction_img"] = self.__get_plot(train_with_test_pred)
        out["rmse"] = sqrt(mean_squared_error(test, train_with_test_pred[-n_test:]))
        return out


def main():
    data = pd.read_csv("data/passengers.csv", header=0, usecols=["passengers"])
    # data = pd.read_csv("data/temperature.csv", header=0, usecols=["Jerusalem"])[:200]
    # data = pd.read_csv("data/temperature.csv", header=0, usecols=["Eilat"])[:400]
    # data = pd.read_csv("data/temperature.csv", header=0, usecols=["Jerusalem"])[:400]
    # data = pd.read_csv("data/temperature.csv", header=0, usecols=["Miami"])[:600]
    # data = pd.read_csv("data/temperature.csv", header=0, usecols=["Tel Aviv District"])[:600]
    # data = pd.read_csv("data/temperature.csv", header=0, usecols=["Jerusalem"])[:700]
    y = data.values
    data = y
    y_node = array.Array(y.shape)
    root_node = copy.Copy(y_node)

    def strange(a, _):
        if a <= 170 and a >= 150:
            return 1729

        return a

    y_node.addfilter(filters.StrangeBehaviour(strange))
    # y_node.addfilter(filters.Gap(prob_break=.1, prob_recover=.5, missing_value=np.nan))

    output = root_node.process(data, np.random.RandomState(seed=42))
    model = Model(output)
    out = model.run()
    out["prediction_img"].show()
    print("RMSE: {}".format(out["rmse"]))


if __name__ == "__main__":
    main()

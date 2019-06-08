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
from keras.preprocessing.sequence import TimeseriesGenerator


class Model:
    def __init__(self, data):
        seed = 42
        rn.seed(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)
        conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        session = tf.Session(graph=tf.get_default_graph(), config=conf)
        backend.set_session(session)
        self.data = data

    def run(self):
        n_input = 1
        n_train = int(len(self.data) * .67)
        n_test = len(self.data) - n_train
        n_steps = min(n_train, 25)
        train = self.data[:n_train]

        train_gen = TimeseriesGenerator(train, train, length=n_steps)
        model = Sequential()
        model.add(LSTM(500, activation="relu", input_shape=(n_steps, n_input)))
        model.add(Dense(1))
        model.compile(loss="mse", optimizer="adam")
        model.fit_generator(train_gen, steps_per_epoch=1, epochs=500)

        predicted_test = train
        x_cur = train[-n_steps:].reshape((1, n_steps, n_input))
        for _ in range(n_test):
            y_cur = model.predict(x_cur)
            predicted_test = np.concatenate([predicted_test, y_cur], axis=0)
            x_cur = np.delete(x_cur, 0, axis=1)
            x_cur = np.concatenate([x_cur, y_cur.reshape((1, 1, n_input))], axis=1)

        plt.plot(self.data, label="data")
        plt.plot(predicted_test, label="pred", zorder=1)
        plt.legend()
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        byte_img = buf.read()
        byte_img = BytesIO(byte_img)

        out = OrderedDict()
        out["prediction_img"] = Image.open(byte_img)
        return out


def main():
    data = pd.read_csv("data/passengers.csv", header=0, usecols=["passengers"]).values
    # data = pd.read_csv("data/temperature.csv", header=0, usecols=["Jerusalem"])[100:300].values
    # data = pd.read_csv("data/temperature.csv", header=0, usecols=["Eilat"])[50:400].values
    # data = pd.read_csv("data/temperature.csv", header=0, usecols=["Tel Aviv District"])[100:500].values
    plt.plot(data)
    plt.tight_layout()
    plt.show()
    model = Model(data)
    out = model.run()
    out["prediction_img"].show()


if __name__ == "__main__":
    main()

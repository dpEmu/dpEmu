import os
import random
from collections import OrderedDict
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from keras.preprocessing.sequence import TimeseriesGenerator


class Model:
    def __init__(self, data):
        os.environ["PYTHONHASHSEED"] = str(42)
        random.seed(42)
        np.random.seed(42)
        tf.set_random_seed(42)
        self.data = data

    def run(self):
        n_features = 1
        n_input = int(len(self.data) * .2)
        n_train = int(len(self.data) * .67)
        n_test = len(self.data) - n_train
        train = self.data[:n_train]

        train_gen = TimeseriesGenerator(train, train, length=n_input)
        model = Sequential()
        model.add(LSTM(200, activation="relu", input_shape=(n_input, n_features)))
        model.add(Dense(1))
        model.compile(loss="mse", optimizer="adam")
        model.fit_generator(train_gen, steps_per_epoch=1, epochs=200, verbose=0)

        predicted_test = train
        x_cur = train[-n_input:].reshape((1, n_input, n_features))
        for _ in range(n_test):
            y_cur = model.predict(x_cur)
            predicted_test = np.concatenate([predicted_test, y_cur], axis=0)
            x_cur = np.delete(x_cur, 0, axis=1)
            x_cur = np.concatenate([x_cur, y_cur.reshape((1, 1, n_features))], axis=1)

        plt.plot(self.data, label="data")
        plt.plot(predicted_test, label="pred")
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
    model = Model(data)
    out = model.run()
    out["prediction_img"].show()


if __name__ == "__main__":
    main()

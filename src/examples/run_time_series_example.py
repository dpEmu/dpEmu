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
        os.environ['PYTHONHASHSEED'] = str(42)
        random.seed(42)
        np.random.seed(42)
        tf.set_random_seed(42)
        self.data = data

    def run(self):
        n_input, n_features = 3, 1
        train_size = int(len(self.data) * .67)
        train, test = self.data[:train_size], self.data[train_size - n_input:]
        train_gen = TimeseriesGenerator(train, train, length=n_input)
        test_gen = TimeseriesGenerator(test, test, length=n_input)

        model = Sequential()
        model.add(LSTM(200, activation="relu", input_shape=(n_input, n_features)))
        model.add(Dense(1))
        model.compile(loss="mse", optimizer="adam")

        model.fit_generator(train_gen, epochs=200, verbose=0)
        predicted_test = model.predict_generator(test_gen)
        print(predicted_test)
        predicted_test = np.concatenate((self.data[:train_size, ], predicted_test))

        plt.plot(self.data)
        plt.plot(predicted_test)
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
    data = pd.read_csv("data/temperature.csv", header=0, usecols=["Vancouver"])[1:91].values
    model = Model(data)
    out = model.run()
    out["prediction_img"].show()


if __name__ == "__main__":
    main()

import numpy as np


class Filter:

    def __init__(self):
        pass


class Missing(Filter):

    def __init__(self, probability):
        self.probability = probability
        super().__init__()

    def apply(self, data):
        mask = np.random.choice([True, False],
                                size=data.shape,
                                p=[self.probability, 1. - self.probability])
        data[mask] = np.nan


class GaussianNoise(Filter):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        super().__init__()

    def apply(self, data):
        data += np.random.normal(loc=self.mean, scale=self.std, size=data.shape)

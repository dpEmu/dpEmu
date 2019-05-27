import numpy as np


class Filter:

    def __init__(self):
        np.random.seed(42)


class Missing(Filter):

    def __init__(self, probability):
        self.probability = probability
        super().__init__()

    def apply(self, data):
        mask = np.random.choice([True, False],
                                size=data.shape,
                                p=[self.probability, 1. - self.probability])
        copy = data.copy()
        copy[mask] = np.nan
        return copy


class GaussianNoise(Filter):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        super().__init__()

    def apply(self, data):
        noise = np.random.normal(loc=self.mean, scale=self.std, size=data.shape)
        return data + noise

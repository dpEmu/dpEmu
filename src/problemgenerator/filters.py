import numpy as np


class Filter:

    def __init__(self):
        np.random.seed(42)
        self.shape = ()


class Missing(Filter):

    def __init__(self, probability):
        self.probability = probability
        super().__init__()

    def apply(self, data, index_tuple):
        mask = np.random.choice([True, False],
                                size=data[index_tuple].shape,
                                p=[self.probability, 1. - self.probability])
        data[index_tuple][mask] = np.nan


class GaussianNoise(Filter):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        super().__init__()

    def apply(self, data, index_tuple):
        data[index_tuple] += np.random.normal(loc=self.mean,
                                              scale=self.std,
                                              size=data[index_tuple].shape)

class Uppercase(Filter):

    def __init__(self, probability):
        self.prob = probability
        super().__init__()

    def apply(self, data, index_tuple):

        def stochastic_upper(char, probability):
            if np.random.binomial(1, probability):
                return char.upper()
            return char

        for index, element in np.ndenumerate(data[index_tuple]):
            original_string = element
            modified_string = "".join([stochastic_upper(c, self.prob) for c in original_string])
            data[index_tuple][index] = modified_string

class MissingArea(Filter):

    def __init__(self, probability, mean_radius, missing_value):
        self.probability = probability
        self.mean_radius = mean_radius
        self.missing_value = missing_value
        super().__init__()

    def apply(self, data, index_tuple):
        pass

import random

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
            modified_string = "".join(
                [stochastic_upper(c, self.prob) for c in original_string])
            data[index_tuple][index] = modified_string


class OCRerror(Filter):

    def __init__(self, replacements):
        """ Pass replacements as a dict.

        For example {"e": (["E", "i"], [.5, .5]), "g": (["q", "9"], [.2, .8])}
        where the latter list consists of probabilities which should sum to 1."""

        random.seed(42)
        self.replacements = replacements
        super().__init__()

    def apply(self, data, index_tuple):
        for index, string_ in np.ndenumerate(data[index_tuple]):
            data[index_tuple][index] = self.generate_ocr_errors(string_)

    def generate_ocr_errors(self, string_):
        return "".join([self.replace_char(c) for c in string_])

    def replace_char(self, c):
        if c in self.replacements:
            chars, probs = self.replacements[c]
            return random.choices(chars, probs)[0]

        return c

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
    class GaussianRadius:
        def __init__(self, mean, std):
            self.mean = mean
            self.std = std

        def generate(self):
            return max(0, self.mean + round(np.random.normal(scale=self.std)))

    class ProbabilityArrayRadius:
        def __init__(self, probability_array):
            self.probability_array = probability_array

        def generate(self):
            sum_of_probabilities = 1
            for radius, _ in enumerate(self.probability_array):
                if np.random.random() <= self.probability_array[radius] / sum_of_probabilities:
                    return radius
                sum_of_probabilities -= self.probability_array[radius]
            return 0 # if for some reason none of the radii is chosen return 0 i.e. no missing area

    def __init__(self, probability, radius_generator, missing_value):
        self.probability = probability
        self.radius_generator = radius_generator
        self.missing_value = missing_value
        super().__init__()

    def apply(self, data, index_tuple):
        for index, _ in np.ndenumerate(data[index_tuple]):
            missing_areas = [] # list of tuples (x, y, radius)

            # generate missing areas
            element = data[index_tuple][index].split("\n")
            for y, _ in enumerate(element):
                for x, _ in enumerate(element[y]):
                    if np.random.random() <= self.probability:
                        missing_areas.append((x, y, self.radius_generator.generate()))

            # replace elements in the missing areas by missing_value
            modified = []
            for y, _ in enumerate(element):
                modified_line = ""
                for x, _ in enumerate(element[y]):
                    inside_missing_area = False
                    for area in missing_areas:
                        if abs(x - area[0]) < area[2] and abs(y - area[1]) < area[2]:
                            inside_missing_area = True
                            break
                    if inside_missing_area:
                        modified_line += self.missing_value
                    else:
                        modified_line += element[y][x]
                modified.append(modified_line)
            data[index_tuple][index] = "\n".join(modified)

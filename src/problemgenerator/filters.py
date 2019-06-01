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

    def __init__(self, probability, mean_radius, std, missing_value):
        self.probability = probability
        self.mean_radius = mean_radius
        self.std = std
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
                        missing_areas.append((x, y, max(0, self.mean_radius + round(np.random.normal(scale=self.std)))))
            
            # replace elements in the missing areas by missing_value
            element = data[index_tuple][index].split("\n")
            modified = []
            for y, _ in enumerate(element):
                modified_line = ""
                for x, _ in enumerate(element[y]):
                    inside_missing_area = False
                    for area in missing_areas:
                        if abs(x - area[0]) <= area[2] and abs(y - area[1]) <= area[2]:
                            inside_missing_area = True
                            break
                    if inside_missing_area:
                        modified_line += self.missing_value
                    else:
                        modified_line += element[y][x]
                modified.append(modified_line)
            data[index_tuple][index] = "\n".join(modified)

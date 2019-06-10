import random

import numpy as np


class Filter:

    def __init__(self):
        np.random.seed(42)
        random.seed(42)
        self.shape = ()


class Missing(Filter):

    def __init__(self, probability):
        self.probability = probability
        super().__init__()

    def apply(self, data, random_state, index_tuple):
        mask = random_state.rand(*(data[index_tuple].shape)) <= self.probability
        data[index_tuple][mask] = np.nan


class GaussianNoise(Filter):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        super().__init__()

    def apply(self, data, random_state, index_tuple):
        data[index_tuple] += random_state.normal(loc=self.mean, scale=self.std, size=data[index_tuple].shape)


class Uppercase(Filter):

    def __init__(self, probability):
        self.prob = probability
        super().__init__()

    def apply(self, data, random_state, index_tuple):

        def stochastic_upper(char, probability):
            if random_state.rand() <= probability:
                return char.upper()
            return char

        for index, element in np.ndenumerate(data[index_tuple]):
            original_string = element
            modified_string = "".join(
                [stochastic_upper(c, self.prob) for c in original_string])
            data[index_tuple][index] = modified_string


class OCRError(Filter):

    def __init__(self, normalized_params, p):
        """ Pass normalized_params as a dict.

        For example {"e": (["E", "i"], [.5, .5]), "g": (["q", "9"], [.2, .8])}
        where the latter list consists of probabilities which should sum to 1."""

        self.normalized_params = normalized_params
        self.p = p
        super().__init__()

    def apply(self, data, random_state, index_tuple):
        for index, string_ in np.ndenumerate(data[index_tuple]):
            data[index_tuple][index] = (self.generate_ocr_errors(string_, random_state))

    def generate_ocr_errors(self, string_, random_state):
        return "".join([self.replace_char(c, random_state) for c in string_])

    def replace_char(self, c, random_state):
        if c in self.normalized_params and random_state.random_sample() < self.p:
            chars, probs = self.normalized_params[c]
            return random_state.choice(chars, 1, p=probs)[0]

        return c


class MissingArea(Filter):
    class GaussianRadiusGenerator:
        def __init__(self, mean, std):
            self.mean = mean
            self.std = std

        def generate(self, random_state):
            return max(0, self.mean + round(random_state.normal(scale=self.std)))

    class ProbabilityArrayRadiusGenerator:
        def __init__(self, probability_array):
            self.probability_array = probability_array

        def generate(self, random_state):
            sum_of_probabilities = 1
            for radius, _ in enumerate(self.probability_array):
                if random_state.random_sample() <= self.probability_array[radius] / sum_of_probabilities:
                    return radius
                sum_of_probabilities -= self.probability_array[radius]
            return 0  # if for some reason none of the radii is chosen return 0 i.e. no missing area

    def __init__(self, probability, radius_generator, missing_value):
        self.probability = probability
        self.radius_generator = radius_generator
        self.missing_value = missing_value
        super().__init__()

    def apply(self, data, random_state, index_tuple):
        def insert_default_value_for_missing_key(key, missing_areas):
            if key not in missing_areas:
                missing_areas[key] = 0

        for index, _ in np.ndenumerate(data[index_tuple]):
            missing_areas = {}  # map with keys (x, y) and values (radius)

            # generate missing areas
            element = data[index_tuple][index].split("\n")
            max_len = 0
            max_radius = 0
            for y, _ in enumerate(element):
                for x, _ in enumerate(element[y]):
                    max_len = max(max_len, x)
                    if random_state.random_sample() <= self.probability:
                        radius = self.radius_generator.generate(random_state)
                        missing_areas[(x - radius, y - radius)] = 2 * radius
                        max_radius = max(max_radius, radius)

            # calculate missing areas
            for y in range(-max_radius, len(element)):
                for x in range(-max_radius, max_len):
                    if (x, y) in missing_areas and missing_areas[(x, y)] > 0:
                        val = missing_areas[(x, y)]
                        insert_default_value_for_missing_key((x + 1, y), missing_areas)
                        insert_default_value_for_missing_key((x, y + 1), missing_areas)
                        insert_default_value_for_missing_key((x + 1, y + 1), missing_areas)
                        missing_areas[(x + 1, y)] = max(missing_areas[(x + 1, y)], val - 1)
                        missing_areas[(x, y + 1)] = max(missing_areas[(x, y + 1)], val - 1)
                        missing_areas[(x + 1, y + 1)] = max(missing_areas[(x + 1, y + 1)], val - 1)

            # replace elements in the missing areas by missing_value
            modified = []
            for y, _ in enumerate(element):
                modified_line = ""
                for x, _ in enumerate(element[y]):
                    if (x, y) in missing_areas:
                        modified_line += self.missing_value
                    else:
                        modified_line += element[y][x]
                modified.append(modified_line)
            data[index_tuple][index] = "\n".join(modified)


class Gap(Filter):
    def __init__(self, prob_break, prob_recover, missing_value=np.nan):
        super().__init__()
        self.missing_value = missing_value
        self.prob_break = prob_break
        self.prob_recover = prob_recover
        self.working = True

    def apply(self, data, random_state, index_tuple):
        """Selects gap lengths from a discrete uniform distribution.

        If a gap just occurred, then enforce a grace period when gaps cannot occur."""
        def update_working_state():
            if self.working:
                if random_state.rand() <= self.prob_break:
                    self.working = False
            else:
                if random_state.rand() <= self.prob_recover:
                    self.working = True

        for index, _ in np.ndenumerate(data[index_tuple]):
            update_working_state()
            if not self.working:
                data[index_tuple][index] = self.missing_value


class SensorDrift(Filter):
    def __init__(self, magnitude):
        """Magnitude is the linear increase in drift during time period t_i -> t_i+1."""
        super().__init__()
        self.magnitude = magnitude
        self.increase = magnitude

    def apply(self, data, random_state, index_tuple):
        for index, _ in np.ndenumerate(data[index_tuple]):
            data[index_tuple][index] += self.increase
            self.increase += self.magnitude


class StrangeBehaviour(Filter):
    def __init__(self, do_strange_behaviour):
        """The function do_strange_behaviour outputs strange sensor values into the data."""
        super().__init__()
        self.do_strange_behaviour = do_strange_behaviour

    def apply(self, data, random_state, index_tuple):
        for index, _ in np.ndenumerate(data[index_tuple]):
            data[index_tuple][index] = self.do_strange_behaviour(data[index_tuple][index], random_state)

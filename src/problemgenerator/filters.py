import random

import numpy as np
from math import pi, sin, cos, sqrt


class Filter:

    def __init__(self):
        np.random.seed(42)
        random.seed(42)
        self.shape = ()


class Missing(Filter):

    def __init__(self, probability):
        self.probability = probability
        super().__init__()

    def apply(self, data, random_state, index_tuple, named_dims):
        mask = random_state.rand(*(data[index_tuple].shape)) <= self.probability
        data[index_tuple][mask] = np.nan


class GaussianNoise(Filter):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        super().__init__()

    def apply(self, data, random_state, index_tuple, named_dims):
        data[index_tuple] += random_state.normal(loc=self.mean, scale=self.std, size=data[index_tuple].shape)


class GaussianNoiseTimeDependent(Filter):
    def __init__(self, mean, std, mean_increase, std_increase):
        self.mean = mean
        self.std = std
        self.mean_increase = mean_increase
        self.std_increase = std_increase

        super().__init__()

    def apply(self, data, random_state, index_tuple, named_dims):
        time = named_dims["time"]
        data[index_tuple] += random_state.normal(loc=self.mean + self.mean_increase * time,
                                                 scale=self.std + self.std_increase * time,
                                                 size=data[index_tuple].shape)


class Uppercase(Filter):

    def __init__(self, probability):
        self.prob = probability
        super().__init__()

    def apply(self, data, random_state, index_tuple, named_dims):

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

    def apply(self, data, random_state, index_tuple, named_dims):
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
    def __init__(self, probability, radius_generator, missing_value):
        self.probability = probability
        self.radius_generator = radius_generator
        self.missing_value = missing_value
        super().__init__()

    def apply(self, data, random_state, index_tuple, named_dims):
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


class StainArea(Filter):
    def __init__(self, probability, radius_generator, transparency_percentage):
        """This filter adds stains to the images.
            probability: probability of adding a stain at each pixel.
            radius_generator: object implementing a generate(random_state) function
                which returns the radius of the stain.
            transparency_percentage: 1 means that the stain is invisible and 0 means
                that the part of the image where the stain is is completely black.
        """
        self.probability = probability
        self.radius_generator = radius_generator
        self.transparency_percentage = transparency_percentage
        super().__init__()

    def apply(self, data, random_state, index_tuple, named_dims):
        for y0, _ in enumerate(data[index_tuple]):
            for x0, _ in enumerate(data[index_tuple][y0]):
                if random_state.random_sample() <= self.probability:
                    radius = self.radius_generator.generate(random_state)
                    for y in range(y0 - radius, y0 + radius + 1):
                        for x in range(x0 - radius, x0 + radius + 1):
                            if y < 0 or x < 0 or y >= len(data[index_tuple]) or x >= len(data[index_tuple][0]):
                                continue
                            v = [data[index_tuple][y][x][0], data[index_tuple][y][x][1], data[index_tuple][y][x][2]]
                            v[0] *= self.transparency_percentage
                            v[1] *= self.transparency_percentage
                            v[2] *= self.transparency_percentage
                            data[index_tuple][y][x] = [round(v[0]), round(v[1]), round(v[2])]


class Gap(Filter):
    def __init__(self, prob_break, prob_recover, missing_value=np.nan):
        super().__init__()
        self.missing_value = missing_value
        self.prob_break = prob_break
        self.prob_recover = prob_recover
        self.working = True

    def apply(self, data, random_state, index_tuple, named_dims):
        def update_working_state():
            if self.working:
                if random_state.rand() < self.prob_break:
                    self.working = False
            else:
                if random_state.rand() < self.prob_recover:
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

    def apply(self, data, random_state, index_tuple, named_dims):
        for index, _ in np.ndenumerate(data[index_tuple]):
            data[index_tuple][index] += self.increase
            self.increase += self.magnitude


class StrangeBehaviour(Filter):
    def __init__(self, do_strange_behaviour):
        """The function do_strange_behaviour outputs strange sensor values into the data."""
        super().__init__()
        self.do_strange_behaviour = do_strange_behaviour

    def apply(self, data, random_state, index_tuple, named_dims):
        for index, _ in np.ndenumerate(data[index_tuple]):
            data[index_tuple][index] = self.do_strange_behaviour(data[index_tuple][index], random_state)


class Rain(Filter):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability

    def apply(self, data, random_state, index_tuple, named_dims):
        width = data[index_tuple].shape[1]
        height = data[index_tuple].shape[0]

        direction = random_state.normal(0, 0.1) * pi + pi / 2

        for y in range(height):
            for x in range(width):
                if random_state.rand() <= self.probability:
                    length = round(random_state.normal(20, 10))
                    for k in range(4):
                        ty = y + sin(direction + pi / 4) * 0.5 * k - length / 2 * sin(direction)
                        tx = x + cos(direction + pi / 4) * 0.5 * k - length / 2 * cos(direction)
                        for _ in range(length):
                            if round(ty) < 0 or round(tx) < 0 or round(ty) >= height or round(tx) >= width:
                                ty += sin(direction)
                                tx += cos(direction)
                                continue
                            brightness = random_state.rand() * 10
                            r = data[round(ty)][round(tx)][0]
                            g = data[round(ty)][round(tx)][1]
                            b = data[round(ty)][round(tx)][2]
                            b = min(b + 30, 255)
                            r = min(r + random_state.normal(brightness, 4), 255)
                            g = min(g + random_state.normal(brightness, 4), 255)
                            b = min(b + random_state.normal(brightness, 4), 255)
                            data[round(ty)][round(tx)] = (round(r), round(g), round(b))
                            ty += sin(direction)
                            tx += cos(direction)


class Snow(Filter):
    def __init__(self, snowflake_probability, snowflake_alpha, snowstorm_alpha):
        super().__init__()
        self.snowflake_probability = snowflake_probability
        self.snowflake_alpha = snowflake_alpha
        self.snowstorm_alpha = snowstorm_alpha

    def apply(self, data, random_state, index_tuple, named_dims):
        def generate_perlin_noise(height, width, random_state):
            # Pierre Vigier's implementation of 2d perlin noise with slight changes.
            # https://github.com/pvigier/perlin-numpy
            #
            # The code is licensed under MIT License:
            #
            # MIT License
            #
            # Copyright (c) 2019 Pierre Vigier
            #
            # Permission is hereby granted, free of charge, to any person obtaining a copy
            # of this software and associated documentation files (the "Software"), to deal
            # in the Software without restriction, including without limitation the rights
            # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
            # copies of the Software, and to permit persons to whom the Software is
            # furnished to do so, subject to the following conditions:
            #
            # The above copyright notice and this permission notice shall be included in all
            # copies or substantial portions of the Software.
            #
            # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
            # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
            # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
            # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
            # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
            # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
            # SOFTWARE.

            def f(t):
                return 6*t**5 - 15*t**4 + 10*t**3

            delta = (1 / height, 1 / width)
            d = (height, width)
            grid = np.mgrid[0:1:delta[0], 0:1:delta[1]].transpose(1, 2, 0) % 1
            # Gradients
            angles = 2 * np.pi * random_state.rand(2, 2)
            gradients = np.dstack((np.cos(angles), np.sin(angles)))
            g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
            g10 = gradients[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
            g01 = gradients[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
            g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)
            # Ramps
            n00 = np.sum(grid * g00, 2)
            n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
            n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1]-1)) * g01, 2)
            n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1]-1)) * g11, 2)
            # Interpolation
            t = f(grid)
            n0 = n00*(1-t[:, :, 0]) + t[:, :, 0]*n10
            n1 = n01*(1-t[:, :, 0]) + t[:, :, 0]*n11
            return np.sqrt(2)*((1-t[:, :, 1])*n0 + t[:, :, 1]*n1)

        def add_noise(data):
            width = data[index_tuple].shape[1]
            height = data[index_tuple].shape[0]
            noise = generate_perlin_noise(height, width, random_state)
            noise = (noise + 1) / 2  # transform the noise to be in range [0, 1]

            # add noise
            for y in range(height):
                for x in range(width):
                    r = data[y][x][0]
                    g = data[y][x][1]
                    b = data[y][x][2]
                    r = int(r + self.snowstorm_alpha * (255 - r) * noise[y][x])
                    g = int(g + self.snowstorm_alpha * (255 - g) * noise[y][x])
                    b = int(b + self.snowstorm_alpha * (255 - b) * noise[y][x])
                    data[y][x] = (r, g, b)

        width = data[index_tuple].shape[1]
        height = data[index_tuple].shape[0]

        # generate snowflakes
        for y in range(height):
            for x in range(width):
                if random_state.rand() <= self.snowflake_probability:
                    radius = round(random_state.normal(5, 2))
                    for tx in range(x - radius, x + radius):
                        for ty in range(y - radius, y + radius):
                            if ty < 0 or tx < 0 or ty >= height or tx >= width:
                                continue
                            r = data[ty][tx][0]
                            g = data[ty][tx][1]
                            b = data[ty][tx][2]
                            dist = sqrt((x - tx) * (x - tx) + (y - ty) * (y - ty))
                            r = round(r + (255 - r) * self.snowflake_alpha * max(0, 1 - dist / radius))
                            g = round(g + (255 - g) * self.snowflake_alpha * max(0, 1 - dist / radius))
                            b = round(b + (255 - b) * self.snowflake_alpha * max(0, 1 - dist / radius))
                            data[ty][tx] = (r, g, b)
        add_noise(data)


class Blur(Filter):
    def __init__(self, repeats):
        super().__init__()
        self.repeats = repeats

    def apply(self, data, random_state, index_tuple, named_dims):
        width = data[index_tuple].shape[1]
        height = data[index_tuple].shape[0]
        for _ in range(self.repeats):
            original = np.copy(data)
            for y0 in range(height):
                for x0 in range(width):
                    pixel_sum = np.array([0, 0, 0])
                    pixel_count = 0
                    for y in range(y0 - 1, y0 + 2):
                        for x in range(x0 - 1, x0 + 2):
                            if y < 0 or x < 0 or y == height or x == width:
                                continue
                            pixel_sum += original[y][x]
                            pixel_count += 1
                    data[y0][x0] = pixel_sum // pixel_count

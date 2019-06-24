import random
from colorsys import rgb_to_hls, hls_to_rgb
from math import pi, sin, cos, sqrt, exp
import numpy as np
from scipy.ndimage import gaussian_filter
import cv2
import imutils
from io import BytesIO
from PIL import Image


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


def replace_inds(mask, str1, str2):
    return "".join([str1[i] if mask[i] else str2[i] for i in range(len(mask))])


class MissingArea(Filter):
    """ TODO: radius_generator is a struct, not a function. It should be a function for repeatability
    """
    def __init__(self, probability, radius_generator, missing_value):
        self.probability = probability
        self.radius_generator = radius_generator
        self.missing_value = missing_value
        super().__init__()

    def apply(self, data, random_state, index_tuple, named_dims):
        if self.probability == 0:
            return

        for index, _ in np.ndenumerate(data[index_tuple]):
            # 1. Get indexes of newline characters. We will not touch those
            string = data[index_tuple][index]

            row_starts = [0]
            for i, c in enumerate(string):
                if c == '\n':
                    row_starts.append(i+1)
            if not row_starts or row_starts[-1] != len(string):
                row_starts.append(len(string))
            height = len(row_starts) - 1

            widths = np.array([row_starts[i+1] - row_starts[i] - 1 for i in range(height)])
            width = np.max(widths)

            # 2. Generate error
            errs = np.zeros(shape=(height+1, width+1))
            ind = -1
            while True:
                ind += random_state.geometric(self.probability)

                if ind >= width * height:
                    break
                y = ind // width
                x = ind - y * width
                r = self.radius_generator.generate(random_state)
                x0 = max(x - r, 0)
                x1 = min(x + r + 1, width)
                y0 = max(y - r, 0)
                y1 = min(y + r + 1, height)
                errs[y0, x0] += 1
                errs[y0, x1] -= 1
                errs[y1, x0] -= 1
                errs[y1, x1] += 1

            # 3. Perform prefix sums, create mask
            errs = np.cumsum(errs, axis=0)
            errs = np.cumsum(errs, axis=1)
            errs = (errs > 0)

            mask = np.zeros(len(string))
            for y in range(height):
                ind = row_starts[y]
                mask[ind:ind + widths[y]] = errs[y, 0:widths[y]]

            # 4. Apply error to string
            res_str = "".join([' ' if mask[i] else string[i] for i in range(len(mask))])
            data[index_tuple][index] = res_str


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
        height = data[index_tuple].shape[0]
        width = data[index_tuple].shape[1]

        # 1. Generate error
        errs = np.zeros(shape=(height+1, width+1))
        ind = -1
        while True:
            ind += random_state.geometric(self.probability)

            if ind >= width * height:
                break
            y = ind // width
            x = ind - y * width
            r = self.radius_generator.generate(random_state)
            x0 = max(x - r, 0)
            x1 = min(x + r + 1, width)
            y0 = max(y - r, 0)
            y1 = min(y + r + 1, height)
            errs[y0, x0] += 1
            errs[y0, x1] -= 1
            errs[y1, x0] -= 1
            errs[y1, x1] += 1

        # 2. Modify the array
        errs = np.cumsum(errs, axis=0)
        errs = np.cumsum(errs, axis=1)
        errs = np.power(self.transparency_percentage, errs)
        for j in range(3):
            data[index_tuple][:, :, j] = np.multiply(data[index_tuple][:, :, j], errs[0:height, 0:width])


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
            # The original code is licensed under MIT License:
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

            d = (height, width)
            grid = np.mgrid[0:d[0], 0:d[1]].astype(float)
            grid[0] /= height
            grid[1] /= width

            grid = grid.transpose(1, 2, 0) % 1
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
            n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
            n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)
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


class JPEG_Compression(Filter):
    """
    Compress the image as JPEG and uncompress. Quality should be in range [1, 100], the bigger the less loss
    """
    def __init__(self, quality):
        super().__init__()
        self.quality = quality

    def apply(self, data, random_state, index_tuple, named_dims):
        iml = Image.fromarray(data)
        buf = BytesIO()
        iml.save(buf, "JPEG", quality=self.quality)
        iml = Image.open(buf)
        res_data = np.array(iml)

        width = data[index_tuple].shape[1]
        height = data[index_tuple].shape[0]
        for y0 in range(height):
            for x0 in range(width):
                data[y0, x0] = res_data[y0, x0]


class Blur_Gaussian(Filter):
    """
    Create blur in images by applying a Gaussian filter.
    The standard deviation of the Gaussian is taken as a parameter.
    """
    def __init__(self, standard_dev):
        super().__init__()
        self.std = standard_dev

    def apply(self, data, random_state, index_tuple, named_dims):
        data[index_tuple] = gaussian_filter(data[index_tuple], self.std)


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


class Resolution(Filter):
    """
    Makes resolution k times smaller. K must be an integer
    """
    def __init__(self, k):
        super().__init__()
        self.k = k

    def apply(self, data, random_state, index_tuple, named_dims):
        width = data[index_tuple].shape[1]
        height = data[index_tuple].shape[0]
        for y0 in range(height):
            y = (y0 // self.k) * self.k
            for x0 in range(width):
                x = (x0 // self.k) * self.k
                data[y0, x0] = data[y, x]


class Rotation(Filter):
    def __init__(self, angle):
        super().__init__()
        self.angle = angle

    def apply(self, data, random_state, index_tuple, named_dims):
        data[index_tuple] = imutils.rotate(data[index_tuple], self.angle)

        # Guesstimation for a large enough resize to avoid black areas in cropped picture
        factor = 1.8
        resized = cv2.resize(data[index_tuple], None, fx=factor, fy=factor)
        resized_width = resized.shape[1]
        resized_height = resized.shape[0]
        width = data[index_tuple].shape[1]
        height = data[index_tuple].shape[0]

        x0 = round((resized_width - width)/2)
        y0 = round((resized_height - height)/2)
        data[index_tuple] = resized[y0:y0+height, x0:x0+width]


class Brightness(Filter):
    """
    Increases or decreases brightness in the image.
    tar: 0 if you want to decrease brightness, 1 if you want to increase it
    rat: scales the brightness change
    """
    def __init__(self, tar, rat):
        super().__init__()
        self.tar = tar
        self.rat = rat

    def apply(self, data, random_state, index_tuple, named_dims):
        width = data[index_tuple].shape[1]
        height = data[index_tuple].shape[0]
        for y0 in range(height):
            for x0 in range(width):
                r = float(data[y0, x0, 0]) * (1 / 255)
                g = float(data[y0, x0, 1]) * (1 / 255)
                b = float(data[y0, x0, 2]) * (1 / 255)
                (hu, li, sa) = rgb_to_hls(r, g, b)

                mult = 1 - exp(-2 * self.rat)
                li = li * (1 - mult) + self.tar * mult

                (r, g, b) = hls_to_rgb(hu, li, sa)
                data[y0, x0, 0] = 255 * r
                data[y0, x0, 1] = 255 * g
                data[y0, x0, 2] = 255 * b


class Saturation(Filter):
    """
    Increases or decreases saturation in the image.
    tar: 0 if you want to decrease saturation, 1 if you want to increase it
    rat: scales the saturation change
    """
    def __init__(self, tar, rat):
        super().__init__()
        self.tar = tar
        self.rat = rat

    def apply(self, data, random_state, index_tuple, named_dims):
        width = data[index_tuple].shape[1]
        height = data[index_tuple].shape[0]
        for y0 in range(height):
            for x0 in range(width):
                r = float(data[y0, x0, 0]) * (1 / 255)
                g = float(data[y0, x0, 1]) * (1 / 255)
                b = float(data[y0, x0, 2]) * (1 / 255)
                (hu, li, sa) = rgb_to_hls(r, g, b)

                mult = 1 - exp(-2 * self.rat * sa)
                sa = sa * (1 - mult) + self.tar * mult

                (r, g, b) = hls_to_rgb(hu, li, sa)
                data[y0, x0, 0] = 255 * r
                data[y0, x0, 1] = 255 * g
                data[y0, x0, 2] = 255 * b


class LensFlare(Filter):
    def __init__(self):
        super().__init__()

    def apply(self, data, random_state, index_tuple, named_dims):
        def flare(x0, y0, radius):
            gt = random_state.randint(130, 180)
            rt = random_state.randint(220, 255)
            bt = random_state.randint(0, 50)
            x_offset = random_state.normal(0, 5)
            y_offset = random_state.normal(0, 5)
            for x in range(x0 - radius, x0 + radius + 1):
                for y in range(y0 - radius, y0 + radius + 1):
                    if y < 0 or x < 0 or y >= height or x >= width:
                        continue
                    dist = sqrt((x - x0) * (x - x0) + (y - y0) * (y - y0))
                    if dist > radius:
                        continue
                    offset_dist = sqrt((x - x0 + x_offset)**2 + (y - y0 + y_offset)**2)
                    r = data[y][x][0]
                    g = data[y][x][1]
                    b = data[y][x][2]
                    a = 3
                    t = max(0, min(1, (1 - (radius - offset_dist) / radius)))
                    visibility = max(0, a * t * t + (1 - a) * t) * 0.8
                    r = round(r + (rt - r) * visibility)
                    g = round(g + (gt - g) * visibility)
                    b = round(b + (bt - b) * visibility)
                    data[y][x] = (r, g, b)

        width = data[index_tuple].shape[1]
        height = data[index_tuple].shape[0]

        # estimate the brightest spot in the image
        pixel_sum_x = [0, 0, 0]
        pixel_sum_y = [0, 0, 0]
        expected_x = [0, 0, 0]
        expected_y = [0, 0, 0]
        for y0 in range(height):
            for x0 in range(width):
                pixel_sum_x += data[y0][x0]
                pixel_sum_y += data[y0][x0]
        for y0 in range(height):
            for x0 in range(width):
                expected_x += x0 * data[y0][x0] / pixel_sum_x
                expected_y += y0 * data[y0][x0] / pixel_sum_y
        best_y = int((expected_y[0] + expected_y[1] + expected_y[2]) / 3)
        best_x = int((expected_x[0] + expected_x[1] + expected_x[2]) / 3)

        origo_vector = np.array([width / 2 - best_x, height / 2 - best_y])
        origo_vector = origo_vector / sqrt(origo_vector[0] * origo_vector[0] + origo_vector[1] * origo_vector[1])

        # move towards origo and draw flares
        y = best_y
        x = best_x
        steps = 0
        while True:
            if steps < 0:
                radius = round(max(40, random_state.normal(100, 100)))
                flare(int(x), int(y), radius)
                steps = random_state.normal(radius, 15)
            if (best_x - width / 2)**2 + (best_y - height / 2)**2 + 1 <= (x - width / 2)**2 + (y - height / 2)**2:
                break
            y += origo_vector[1]
            x += origo_vector[0]
            steps -= 1


class ApplyWithProbability(Filter):
    def __init__(self, ftr, probability):
        super().__init__()
        self.ftr = ftr
        self.probability = probability

    def apply(self, data, random_state, index_tuple, named_dims):
        if random_state.rand() < self.probability:
            self.ftr.apply(data, random_state, index_tuple, named_dims)


class Constant(Filter):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def apply(self, data, random_state, index_tuple, named_dims):
        data[index_tuple].fill(self.value)


class Identity(Filter):
    def __init__(self):
        super().__init__()

    def apply(self, data, random_state, index_tuple, named_dims):
        pass


class BinaryFilter(Filter):
    def __init__(self, filter_a, filter_b):
        super().__init__()
        self.filter_a = filter_a
        self.filter_b = filter_b

    def apply(self, data, random_state, index_tuple, named_dims):
        data_a = data.copy()
        data_b = data.copy()
        self.filter_a.apply(data_a, random_state, index_tuple, named_dims)
        self.filter_b.apply(data_b, random_state, index_tuple, named_dims)
        for index, _ in np.ndenumerate(data[index_tuple]):
            data[index] = self.operation(data_a[index], data_b[index])

    def operation(self, element_a, element_b):
        raise NotImplementedError()


class Addition(BinaryFilter):
    def operation(self, element_a, element_b):
        return element_a + element_b


class Subtraction(BinaryFilter):
    def operation(self, element_a, element_b):
        return element_a - element_b


class Multiplication(BinaryFilter):
    def operation(self, element_a, element_b):
        return element_a * element_b


class Division(BinaryFilter):
    def operation(self, element_a, element_b):
        return element_a / element_b


class IntegerDivision(BinaryFilter):
    def operation(self, element_a, element_b):
        return element_a // element_b


class Modulo(BinaryFilter):
    def operation(self, element_a, element_b):
        return element_a % element_b


class And(BinaryFilter):
    def operation(self, element_a, element_b):
        return element_a & element_b


class Or(BinaryFilter):
    def operation(self, element_a, element_b):
        return element_a | element_b


class Xor(BinaryFilter):
    def operation(self, element_a, element_b):
        return element_a ^ element_b


class Difference(Filter):
    """
    Returns the difference between the original and the filtered data,
    i.e. it is shorthand for Subtraction(filter, Identity()).
    """
    def __init__(self, ftr):
        super().__init__()
        self.ftr = Subtraction(ftr, Identity())

    def apply(self, data, random_state, index_tuple, named_dims):
        self.ftr.apply(data, random_state, index_tuple, named_dims)


class Max(BinaryFilter):
    def operation(self, element_a, element_b):
        return max(element_a, element_b)


class Min(BinaryFilter):
    def operation(self, element_a, element_b):
        return min(element_a, element_b)

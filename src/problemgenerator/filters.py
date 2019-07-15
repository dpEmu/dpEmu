import random
from colorsys import rgb_to_hls, hls_to_rgb
from math import pi, sin, cos, sqrt
import numpy as np
from scipy.ndimage import gaussian_filter
import cv2
import imutils
from io import BytesIO
from PIL import Image


class Filter:
    """A Filter is an error source which can be attached to an Array node.
    The apply method applies the filter. A filter may always assume that
    it is acting upon a Numpy array. (When the underlying data object is not
    a Numpy array, the required conversions are performed by the Array node
    to which the Filter is attached.)
    """

    def __init__(self):
        np.random.seed(42)
        random.seed(42)
        self.shape = ()

    def set_params(self, params_dict):
        pass


class Missing(Filter):
    """For each element in the array, change the value of the element to nan
    with the provided probability.
    """

    def __init__(self, probability_id):
        self.probability_id = probability_id
        super().__init__()

    def set_params(self, params_dict):
        self.probability = params_dict[self.probability_id]

    def apply(self, node_data, random_state, named_dims):
        mask = random_state.rand(*(node_data.shape)) <= self.probability
        node_data[mask] = np.nan


class Clip(Filter):
    """Clip values to minimum and maximum value provided by the user.
    """
    def __init__(self, min_id, max_id):
        self.min_id = min_id
        self.max_id = max_id
        super().__init__()

    def set_params(self, params_dict):
        self.min = params_dict[self.min_id]
        self.max = params_dict[self.max_id]

    def apply(self, node_data, random_state, named_dims):
        np.clip(node_data, self.min, self.max, out=node_data)


class GaussianNoise(Filter):
    """For each element in the array add noise drawn from a Gaussian distribution
    with the provided parameters mean and std (standard deviation).
    """
    def __init__(self, mean_id, std_id):
        self.mean_id = mean_id
        self.std_id = std_id
        super().__init__()

    def set_params(self, params_dict):
        self.mean = params_dict[self.mean_id]
        self.std = params_dict[self.std_id]

    def apply(self, node_data, random_state, named_dims):
        node_data += random_state.normal(loc=self.mean,
                                         scale=self.std,
                                         size=node_data.shape)


class GaussianNoiseTimeDependent(Filter):
    """For each element in the array add noise drawn from a Gaussian distribution
    with the provided parameters mean and std (standard deviation). The mean and
    standard deviation increase with every unit of time by the amount specified
    in the last two parameters.
    """
    def __init__(self, mean_id, std_id, mean_increase_id, std_increase_id):
        self.mean_id = mean_id
        self.std_id = std_id
        self.mean_increase_id = mean_increase_id
        self.std_increase_id = std_increase_id
        super().__init__()

    def set_params(self, params_dict):
        self.mean = params_dict[self.mean_id]
        self.mean_increase = params_dict[self.mean_increase_id]
        self.std = params_dict[self.std_id]
        self.std_increase = params_dict[self.std_increase_id]

    def apply(self, node_data, random_state, named_dims):
        time = named_dims["time"]
        node_data += random_state.normal(loc=self.mean + self.mean_increase * time,
                                         scale=self.std + self.std_increase * time,
                                         size=node_data.shape)


class Uppercase(Filter):
    """For each character in the string, convert the character
    to uppercase with the provided probability.
    """
    def __init__(self, probability_id):
        self.prob_id = probability_id
        super().__init__()

    def set_params(self, params_dict):
        self.prob = params_dict[self.prob_id]

    def apply(self, node_data, random_state, named_dims):

        def stochastic_upper(char, probability):
            if random_state.rand() <= probability:
                return char.upper()
            return char

        for index, element in np.ndenumerate(node_data):
            original_string = element
            modified_string = "".join(
                [stochastic_upper(c, self.prob) for c in original_string])
            node_data[index] = modified_string


class OCRError(Filter):

    def __init__(self, normalized_params_id, p_id):
        """ Pass normalized_params as a dict.

        For example {"e": (["E", "i"], [.5, .5]), "g": (["q", "9"], [.2, .8])}
        where the latter list consists of probabilities which should sum to 1."""

        self.normalized_params_id = normalized_params_id
        self.p_id = p_id
        super().__init__()

    def set_params(self, params_dict):
        self.normalized_params = params_dict[self.normalized_params_id]
        self.p = params_dict[self.p_id]

    def apply(self, node_data, random_state, named_dims):
        for index, string_ in np.ndenumerate(node_data):
            node_data[index] = (self.generate_ocr_errors(string_, random_state))

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
    def __init__(self, probability_id, radius_generator_id, missing_value_id):
        self.probability_id = probability_id
        self.radius_generator_id = radius_generator_id
        self.missing_value_id = missing_value_id
        super().__init__()

    def set_params(self, params_dict):
        self.probability = params_dict[self.probability_id]
        self.radius_generator = params_dict[self.radius_generator_id]
        self.missing_value = params_dict[self.missing_value_id]

    def apply(self, node_data, random_state, named_dims):
        if self.probability == 0:
            return

        for index, _ in np.ndenumerate(node_data):
            # 1. Get indexes of newline characters. We will not touch those
            string = node_data[index]

            row_starts = [0]
            for i, c in enumerate(string):
                if c == '\n':
                    row_starts.append(i+1)
            if not row_starts or row_starts[-1] != len(string):
                row_starts.append(len(string))
            height = len(row_starts) - 1

            widths = np.array([row_starts[i+1] - row_starts[i] - 1 for i in range(height)])
            if len(widths) > 0:
                width = np.max(widths)
            else:
                width = 0

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
            node_data[index] = res_str


class StainArea(Filter):
    def __init__(self, probability_id, radius_generator_id, transparency_percentage_id):
        """This filter adds stains to the images.
            probability: probability of adding a stain at each pixel.
            radius_generator: object implementing a generate(random_state) function
                which returns the radius of the stain.
            transparency_percentage: 1 means that the stain is invisible and 0 means
                that the part of the image where the stain is is completely black.
        """
        self.probability_id = probability_id
        self.radius_generator_id = radius_generator_id
        self.transparency_percentage_id = transparency_percentage_id
        super().__init__()

    def set_params(self, params_dict):
        self.probability = params_dict[self.probability_id]
        self.radius_generator = params_dict[self.radius_generator_id]
        self.transparency_percentage = params_dict[self.transparency_percentage_id]

    def apply(self, node_data, random_state, named_dims):
        height = node_data.shape[0]
        width = node_data.shape[1]

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
            node_data[:, :, j] = np.multiply(node_data[:, :, j], errs[0:height, 0:width])


class Gap(Filter):
    def __init__(self, prob_break_id, prob_recover_id, missing_value_id):
        super().__init__()
        self.prob_break_id = prob_break_id
        self.prob_recover_id = prob_recover_id
        self.missing_value_id = missing_value_id
        self.working = True

    def set_params(self, params_dict):
        self.prob_break = params_dict[self.prob_break_id]
        self.prob_recover = params_dict[self.prob_recover_id]
        self.missing_value = params_dict[self.missing_value_id]
        self.working = True

    def apply(self, node_data, random_state, named_dims):
        def update_working_state():
            if self.working:
                if random_state.rand() < self.prob_break:
                    self.working = False
            else:
                if random_state.rand() < self.prob_recover:
                    self.working = True

        # random_state.rand(node_data.shape[0], node_data.shape[1])
        for index, _ in np.ndenumerate(node_data):
            update_working_state()
            if not self.working:
                node_data[index] = self.missing_value


class SensorDrift(Filter):
    def __init__(self, magnitude_id):
        """Magnitude is the linear increase in drift during time period t_i -> t_i+1."""
        super().__init__()
        self.magnitude_id = magnitude_id

    def set_params(self, params_dict):
        self.magnitude = params_dict[self.magnitude_id]

    def apply(self, node_data, random_state, named_dims):
        increases = np.arange(1, node_data.shape[0] + 1) * self.magnitude
        node_data += increases.reshape(node_data.shape)


class StrangeBehaviour(Filter):
    def __init__(self, do_strange_behaviour_id):
        """The function do_strange_behaviour outputs strange sensor values into the data."""
        super().__init__()
        self.do_strange_behaviour_id = do_strange_behaviour_id

    def set_params(self, params_dict):
        self.do_strange_behaviour = params_dict[self.do_strange_behaviour_id]

    def apply(self, node_data, random_state, named_dims):
        for index, _ in np.ndenumerate(node_data):
            node_data[index] = self.do_strange_behaviour(node_data[index], random_state)


class Rain(Filter):
    def __init__(self, probability_id):
        super().__init__()
        self.probability_id = probability_id

    def set_params(self, params_dict):
        self.probability = params_dict[self.probability_id]

    def apply(self, node_data, random_state, named_dims):
        width = node_data.shape[1]
        height = node_data.shape[0]

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
                            r = node_data[round(ty)][round(tx)][0]
                            g = node_data[round(ty)][round(tx)][1]
                            b = node_data[round(ty)][round(tx)][2]
                            b = min(b + 30, 255)
                            r = min(r + random_state.normal(brightness, 4), 255)
                            g = min(g + random_state.normal(brightness, 4), 255)
                            b = min(b + random_state.normal(brightness, 4), 255)
                            node_data[round(ty)][round(tx)] = (round(r), round(g), round(b))
                            ty += sin(direction)
                            tx += cos(direction)


class Snow(Filter):
    def __init__(self, snowflake_probability_id, snowflake_alpha_id, snowstorm_alpha_id):
        super().__init__()
        self.snowflake_probability_id = snowflake_probability_id
        self.snowflake_alpha_id = snowflake_alpha_id
        self.snowstorm_alpha_id = snowstorm_alpha_id

    def set_params(self, params_dict):
        self.snowflake_probability = params_dict[self.snowflake_probability_id]
        self.snowflake_alpha = params_dict[self.snowflake_alpha_id]
        self.snowstorm_alpha = params_dict[self.snowstorm_alpha_id]

    def apply(self, node_data, random_state, named_dims):
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

        def build_snowflake(r):
            res = np.zeros(shape=(2*r+1, 2*r+1))
            for y in range(0, 2*r+1):
                for x in range(0, 2*r+1):
                    dy = y - r
                    dx = x - r
                    d = sqrt(dx*dx + dy*dy)
                    if r == 0:
                        res[y, x] = 1
                    else:
                        res[y, x] = max(0, 1 - d / r)
            return res * self.snowflake_alpha

        width = node_data.shape[1]
        height = node_data.shape[0]

        # generate snowflakes
        flakes = []
        ind = -1
        while True:
            ind += random_state.geometric(self.snowflake_probability)
            if ind >= height * width:
                break
            y = ind // width
            x = ind % width
            r = round(random_state.normal(5, 2))
            if r <= 0:
                continue
            while len(flakes) <= r:
                flakes.append(build_snowflake(len(flakes)))
            y0 = max(0, y-r)
            x0 = max(0, x-r)
            y1 = min(height-1, y+r)+1
            x1 = min(width-1, x+r)+1
            fy0 = y0-(y-r)
            fx0 = x0-(x-r)
            fy1 = y1-(y-r)
            fx1 = x1-(x-r)
            for j in range(3):
                node_data[y0:y1, x0:x1, j] += ((255 - node_data[y0:y1, x0:x1, j]) *
                                               flakes[r][fy0:fy1, fx0:fx1]).astype(int)

        # add noise
        noise = generate_perlin_noise(height, width, random_state)
        noise = (noise + 1) / 2  # transform the noise to be in range [0, 1]
        for j in range(3):
            node_data[:, :, j] += (self.snowstorm_alpha * (255 - node_data[:, :, j]) * noise[:, :]).astype(int)


class JPEG_Compression(Filter):
    """
    Compress the image as JPEG and uncompress. Quality should be in range [1, 100], the bigger the less loss
    """
    def __init__(self, quality_id):
        super().__init__()
        self.quality_id = quality_id

    def set_params(self, params_dict):
        self.quality = params_dict[self.quality_id]

    def apply(self, node_data, random_state, named_dims):
        iml = Image.fromarray(node_data)
        buf = BytesIO()
        iml.save(buf, "JPEG", quality=self.quality)
        iml = Image.open(buf)
        res_data = np.array(iml)

        # width = node_data.shape[1]
        # height = node_data.shape[0]
        node_data[:, :] = res_data


class Blur_Gaussian(Filter):
    """
    Create blur in images by applying a Gaussian filter.
    The standard deviation of the Gaussian is taken as a parameter.
    """
    def __init__(self, standard_dev_id):
        super().__init__()
        self.std_id = standard_dev_id

    def set_params(self, params_dict):
        self.std = params_dict[self.std_id]

    def apply(self, node_data, random_state, named_dims):
        node_data = gaussian_filter(node_data, self.std)


class Blur(Filter):

    def __init__(self, repeats_id):
        super().__init__()
        self.repeats_id = repeats_id

    def set_params(self, params_dict):
        self.repeats = params_dict[self.repeats_id]

    def apply(self, node_data, random_state, named_dims):
        width = node_data.shape[1]
        height = node_data.shape[0]
        for _ in range(self.repeats):
            original = np.copy(node_data)
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
                    node_data[y0][x0] = pixel_sum // pixel_count


class Resolution(Filter):
    """
    Makes resolution k times smaller. K must be an integer
    """
    def __init__(self, k_id):
        super().__init__()
        self.k_id = k_id

    def set_params(self, params_dict):
        self.k = params_dict[self.k_id]

    def apply(self, node_data, random_state, named_dims):
        width = node_data.shape[1]
        height = node_data.shape[0]
        for y0 in range(height):
            y = (y0 // self.k) * self.k
            for x0 in range(width):
                x = (x0 // self.k) * self.k
                node_data[y0, x0] = node_data[y, x]


class ResolutionVectorized(Filter):
    """
    Makes resolution k times smaller. K must be an integer
    """
    def __init__(self, k_id):
        super().__init__()
        self.k_id = k_id

    def set_params(self, params_dict):
        self.k = params_dict[self.k_id]

    def apply(self, node_data, random_state, named_dims):
        w = node_data.shape[1]
        h = node_data.shape[0]
        row, col = (np.indices((h, w)) // self.k) * self.k
        node_data[...] = node_data[row, col]


class Rotation(Filter):
    def __init__(self, angle_id):
        super().__init__()
        self.angle_id = angle_id

    def set_params(self, params_dict):
        self.angle = params_dict[self.angle_id]

    def apply(self, node_data, random_state, named_dims):
        node_data = imutils.rotate(node_data, self.angle)

        # Guesstimation for a large enough resize to avoid black areas in cropped picture
        factor = 1.8
        resized = cv2.resize(node_data, None, fx=factor, fy=factor)
        resized_width = resized.shape[1]
        resized_height = resized.shape[0]
        width = node_data.shape[1]
        height = node_data.shape[0]

        x0 = round((resized_width - width)/2)
        y0 = round((resized_height - height)/2)
        node_data = resized[y0:y0+height, x0:x0+width]


class Brightness(Filter):
    """
    Increases or decreases brightness in the image.
    tar: 0 if you want to decrease brightness, 1 if you want to increase it
    rat: scales the brightness change
    range_id: RGB values are presented either in the range [0,1]
            or in the set {0,...,255}
    """

    def __init__(self, tar_id, rat_id, range_id):
        super().__init__()
        self.tar_id = tar_id
        self.rat_id = rat_id
        self.range_id = range_id

    def set_params(self, params_dict):
        self.tar = params_dict[self.tar_id]
        self.rat = params_dict[self.rat_id]
        # self.range should have value 1 or 255
        self.range = params_dict[self.range_id]

    def apply(self, node_data, random_state, named_dims):
        width = node_data.shape[1]
        height = node_data.shape[0]
        for y0 in range(height):
            for x0 in range(width):
                r = float(node_data[y0, x0, 0]) * (1 / self.range)
                g = float(node_data[y0, x0, 1]) * (1 / self.range)
                b = float(node_data[y0, x0, 2]) * (1 / self.range)
                (hu, li, sa) = rgb_to_hls(r, g, b)

                mult = 1 - np.exp(-2 * self.rat)
                li = li * (1 - mult) + self.tar * mult

                (r, g, b) = hls_to_rgb(hu, li, sa)
                node_data[y0, x0, 0] = self.range * r
                node_data[y0, x0, 1] = self.range * g
                node_data[y0, x0, 2] = self.range * b


class BrightnessVectorized(Filter):
    """
    Increases or decreases brightness in the image.
    tar: 0 if you want to decrease brightness, 1 if you want to increase it
    rat: scales the brightness change
    range_id: RGB values are presented either in the range [0,1]
            or in the set {0,...,255}
    """

    def __init__(self, tar_id, rat_id, range_id):
        super().__init__()
        self.tar_id = tar_id
        self.rat_id = rat_id
        self.range_id = range_id

    def set_params(self, params_dict):
        self.tar = params_dict[self.tar_id]
        self.rat = params_dict[self.rat_id]
        # self.range should have value 1 or 255
        self.range = params_dict[self.range_id]

    def apply(self, node_data, random_state, named_dims):
        nd = node_data.astype("float32")
        if self.range == 255:
            nd[...] = node_data * (1 / self.range)

        hls = cv2.cvtColor(nd, cv2.COLOR_RGB2HLS)
        mult = 1 - np.exp(-2 * self.rat)
        hls[:, :, 1] = hls[:, :, 1] * (1 - mult) + self.tar * mult
        nd[...] = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)

        if self.range == 255:
            nd[...] = nd * self.range
            nd = nd.astype(np.int8)
        else:
            nd = np.clip(nd, 0.0, 1.0)

        node_data[...] = nd


class Saturation(Filter):
    """
    Increases or decreases brightness in the image.
    tar: 0 if you want to decrease brightness, 1 if you want to increase it
    rat: scales the brightness change
    range_id: RGB values are presented either in the range [0,1]
            or in the discrete set {0,...,255}
    """

    def __init__(self, tar_id, rat_id, range_id):
        super().__init__()
        self.tar_id = tar_id
        self.rat_id = rat_id
        self.range_id = range_id

    def set_params(self, params_dict):
        self.tar = params_dict[self.tar_id]
        self.rat = params_dict[self.rat_id]
        # self.range should have value 1 or 255
        self.range = params_dict[self.range_id]

    def apply(self, node_data, random_state, named_dims):
        width = node_data.shape[1]
        height = node_data.shape[0]
        for y0 in range(height):
            for x0 in range(width):
                r = float(node_data[y0, x0, 0]) * (1 / self.range)
                g = float(node_data[y0, x0, 1]) * (1 / self.range)
                b = float(node_data[y0, x0, 2]) * (1 / self.range)
                (hu, li, sa) = rgb_to_hls(r, g, b)

                mult = 1 - np.exp(-2 * self.rat * sa)
                sa = sa * (1 - mult) + self.tar * mult

                (r, g, b) = hls_to_rgb(hu, li, sa)
                node_data[y0, x0, 0] = self.range * r
                node_data[y0, x0, 1] = self.range * g
                node_data[y0, x0, 2] = self.range * b


class SaturationVectorized(Filter):
    """
    Increases or decreases brightness in the image.
    tar: 0 if you want to decrease brightness, 1 if you want to increase it
    rat: scales the brightness change
    range_id: RGB values are presented either in the range [0,1]
            or in the discrete set {0,...,255}
    """

    def __init__(self, tar_id, rat_id, range_id):
        super().__init__()
        self.tar_id = tar_id
        self.rat_id = rat_id
        self.range_id = range_id

    def set_params(self, params_dict):
        self.tar = params_dict[self.tar_id]
        self.rat = params_dict[self.rat_id]
        # self.range should have value 1 or 255
        self.range = params_dict[self.range_id]

    def apply(self, node_data, random_state, named_dims):
        nd = node_data.astype("float32")
        if self.range == 255:
            nd[...] = node_data * (1 / self.range)

        hls = cv2.cvtColor(nd, cv2.COLOR_RGB2HLS)
        mult = 1 - np.exp(-2 * self.rat * hls[:, :, 2])
        hls[:, :, 2] = hls[:, :, 2] * (1 - mult) + self.tar * mult
        nd[...] = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)

        if self.range == 255:
            nd[...] = nd * self.range
            nd = nd.astype(np.int8)
        else:
            nd = np.clip(nd, 0.0, 1.0)

        node_data[...] = nd


class LensFlare(Filter):
    def __init__(self):
        super().__init__()

    def apply(self, node_data, random_state, named_dims):
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
                    r = node_data[y][x][0]
                    g = node_data[y][x][1]
                    b = node_data[y][x][2]
                    a = 3
                    t = max(0, min(1, (1 - (radius - offset_dist) / radius)))
                    visibility = max(0, a * t * t + (1 - a) * t) * 0.8
                    r = round(r + (rt - r) * visibility)
                    g = round(g + (gt - g) * visibility)
                    b = round(b + (bt - b) * visibility)
                    node_data[y][x] = (r, g, b)

        width = node_data.shape[1]
        height = node_data.shape[0]

        # estimate the brightest spot in the image
        pixel_sum_x = [0, 0, 0]
        pixel_sum_y = [0, 0, 0]
        expected_x = [0, 0, 0]
        expected_y = [0, 0, 0]
        for y0 in range(height):
            for x0 in range(width):
                pixel_sum_x += node_data[y0][x0]
                pixel_sum_y += node_data[y0][x0]
        for y0 in range(height):
            for x0 in range(width):
                expected_x += x0 * node_data[y0][x0] / pixel_sum_x
                expected_y += y0 * node_data[y0][x0] / pixel_sum_y
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
    def __init__(self, ftr_id, probability_id):
        super().__init__()
        self.ftr_id = ftr_id
        self.probability_id = probability_id

    def set_params(self, params_dict):
        self.ftr = params_dict[self.ftr_id]
        self.probability = params_dict[self.probability_id]

    def apply(self, node_data, random_state, named_dims):
        if random_state.rand() < self.probability:
            self.ftr.apply(node_data, random_state, named_dims)


class Constant(Filter):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def apply(self, node_data, random_state, named_dims):
        node_data.fill(self.value)


class Identity(Filter):
    def __init__(self):
        super().__init__()

    def apply(self, node_data, random_state, named_dims):
        pass


class BinaryFilter(Filter):
    def __init__(self, filter_a, filter_b):
        super().__init__()
        self.filter_a = filter_a
        self.filter_b = filter_b

    def apply(self, node_data, random_state, named_dims):
        data_a = node_data.copy()
        data_b = node_data.copy()
        self.filter_a.apply(data_a, random_state, named_dims)
        self.filter_b.apply(data_b, random_state, named_dims)
        for index, _ in np.ndenumerate(node_data):
            node_data[index] = self.operation(data_a[index], data_b[index])

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

    def apply(self, node_data, random_state, named_dims):
        self.ftr.apply(node_data, random_state, named_dims)


class Max(BinaryFilter):
    def operation(self, element_a, element_b):
        return max(element_a, element_b)


class Min(BinaryFilter):
    def operation(self, element_a, element_b):
        return min(element_a, element_b)


class ModifyAsDataType(Filter):
    def __init__(self, dtype, ftr):
        super().__init__()
        self.dtype = dtype
        self.ftr = ftr

    def apply(self, node_data, random_state, named_dims):
        copy = node_data.copy().astype(self.dtype)
        self.ftr.apply(copy, random_state, named_dims)
        copy = copy.astype(node_data.dtype)
        for index, _ in np.ndenumerate(node_data):
            node_data[index] = copy[index]

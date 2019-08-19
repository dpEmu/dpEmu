from io import BytesIO
from math import sqrt, sin, cos, pi
import cv2
import numpy as np
import imutils
from PIL import Image
from scipy.ndimage import gaussian_filter
from dpemu.filters import Filter


class Blur(Filter):
    def __init__(self, repeats_id, radius_id=None):
        super().__init__()
        self.repeats_id = repeats_id
        self.radius_id = radius_id

    def set_params(self, params_dict):
        self.repeats = params_dict[self.repeats_id]
        if self.radius_id is not None:
            self.radius = params_dict[self.radius_id]
        else:
            self.radius = 1

    def apply(self, node_data, random_state, named_dims):
        def avg(radius, data):
            height = data.shape[0]
            width = data.shape[1]
            diam = 2*radius + 1
            temp = np.zeros(shape=(height + diam, width + diam))
            temp[0:height, 0:width] += data
            temp[diam:height + diam, 0:width] -= data
            temp[0:height, diam:width + diam] -= data
            temp[diam:height + diam, diam:width + diam] += data
            temp = np.cumsum(temp, axis=0)
            temp = np.cumsum(temp, axis=1)
            return temp[radius:height + radius, radius:width + radius]

        ones = np.ones(shape=(node_data.shape[0], node_data.shape[1]))
        div = avg(self.radius, ones)
        for _ in range(self.repeats):
            if len(node_data.shape) == 2:
                node_data[:, :] = avg(self.radius, node_data) // div
            else:
                for i in range(node_data.shape[-1]):
                    node_data[:, :, i] = avg(self.radius, node_data[:, :, i]) // div


class Resolution(Filter):
    """Makes resolution k times smaller.

    Resolution is changed with the formula:

    new_image[y][x] = image[k * (y // k)][k * (x // k)] for all y and x,

    where // is Python's integer division. K must be an integer.

    Inherits Filter class.
    """

    def __init__(self, k_id):
        """
        Args:
            k_id (str): A key which maps to the k value.
        """
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
    """Rotates the filter.

    If only min_angle is provided, the the image is rotated according to the angle.
    If both min_angle and max_angle are provided, then the rotation angle is chosen
    randomly from the uniform distribution Uniform(min_angle, max_angle).

    If the angle is positive, then the image is rotated counterclockwise.
    Otherwise, the image is rotated clockwise.

    Inherits Filter class.
    """

    def __init__(self, min_angle_id, max_angle_id=None):
        super().__init__()
        self.min_angle_id = min_angle_id
        if max_angle_id is not None:
            self.max_angle_id = max_angle_id
        else:
            self.max_angle_id = min_angle_id

    def set_params(self, params_dict):
        self.min_angle = params_dict[self.min_angle_id]
        self.max_angle = params_dict[self.max_angle_id]

    def apply(self, node_data, random_state, named_dims):
        # Randomize angle, calculate optimal scale ratio
        angle = random_state.uniform(self.min_angle, self.max_angle)
        width = node_data.shape[1]
        height = node_data.shape[0]
        ra = abs(angle % 180) * pi/180
        ra = min(ra, pi - ra)
        factor = sin(ra) * max(width, height) / min(width, height) + cos(ra)

        node_data[...] = imutils.rotate(node_data, angle)
        resized = cv2.resize(node_data, None, fx=factor, fy=factor)
        resized_width = resized.shape[1]
        resized_height = resized.shape[0]

        x0 = round((resized_width - width) / 2)
        y0 = round((resized_height - height) / 2)
        node_data[...] = resized[y0:y0 + height, x0:x0 + width]


class Brightness(Filter):
    """Increases or decreases brightness in the image.

    tar: 0 if you want to decrease brightness, 1 if you want to increase it.

    rat: scales the brightness change.

    range: Should have value 1 or 255. The value is chosen according to how RGB values are presented in
    the corresponding NumPy array. Normally the values are either in the range [0,1] or in the set
    {0,...,255}. If this value is chosen incorrectly, then the filter will produce undesired
    effects on the image.

    Inherits Filter class.
    """

    def __init__(self, tar_id, rat_id, range_id):
        """
        Args:
            tar_id (str): A key which maps to the tar value.
            rat_id (str): A key which maps to the rat value.
            range_id (str): A key which maps to the range value.
        """
        super().__init__()
        self.tar_id = tar_id
        self.rat_id = rat_id
        self.range_id = range_id

    def set_params(self, params_dict):
        self.tar = params_dict[self.tar_id]
        self.rat = params_dict[self.rat_id]
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
            nd = nd.astype(np.uint8)
        else:
            nd = np.clip(nd, 0.0, 1.0)

        node_data[...] = nd


class BlurGaussian(Filter):
    """Blur image according to a zero-centred normal distribution.

    Create blur in images by applying a Gaussian filter.
    The standard deviation of the Gaussian is taken as a parameter.

    Inherits Filter class.
    """

    def __init__(self, standard_dev_id):
        """
        Args:
            standard_dev_id (str): A key which maps to standard deviation.
        """
        super().__init__()
        self.std_id = standard_dev_id

    def set_params(self, params_dict):
        self.std = params_dict[self.std_id]

    def apply(self, node_data, random_state, named_dims):
        if len(node_data.shape) == 2:
            node_data[...] = gaussian_filter(node_data, self.std)
        else:
            for i in range(node_data.shape[-1]):
                node_data[:, :, i] = gaussian_filter(node_data[:, :, i], self.std)


class JPEG_Compression(Filter):
    """Compresses a JPEG-image.

    Compress the image as JPEG and uncompress. Quality should be in range [1, 100],
    the bigger the less loss.

    Inherits Filter class.
    """

    def __init__(self, quality_id):
        super().__init__()
        self.quality_id = quality_id

    def set_params(self, params_dict):
        self.quality = params_dict[self.quality_id]

    def apply(self, node_data, random_state, named_dims):
        iml = Image.fromarray(np.uint8(np.around(node_data)))
        buf = BytesIO()
        iml.save(buf, "JPEG", quality=self.quality)
        iml = Image.open(buf)
        res_data = np.array(iml)

        # width = node_data.shape[1]
        # height = node_data.shape[0]
        node_data[:, :] = res_data


class Rain(Filter):
    """Add rain to images.

    RGB values are presented either in the range [0,1] or in the set {0,...,255},
        thus range should either have value 1 or value 255.

    Inherits Filter class.
    """

    def __init__(self, probability_id, range_id):
        """
        Args:
            probability_id (str): A key which maps to a probability of rain.
            range_id (str): A key which maps to value of either 1 or 255.
        """
        super().__init__()
        self.probability_id = probability_id
        self.range_id = range_id

    def set_params(self, params_dict):
        self.probability = params_dict[self.probability_id]
        # self.range should have value 1 or 255
        self.range = params_dict[self.range_id]

    def apply(self, node_data, random_state, named_dims):
        height = node_data.shape[0]
        width = node_data.shape[1]

        # 1. Generate error
        errs = np.zeros(shape=(height + 1, width + 1))
        ind = -1
        while True:
            ind += random_state.geometric(self.probability)

            if ind >= width * height:
                break
            y = ind // width
            x = ind - y * width
            x_r = 1
            y_r = max(0, round(random_state.normal(20, 10)))
            x0 = max(x - x_r, 0)
            x1 = min(x + x_r + 1, width)
            y0 = max(y - y_r, 0)
            y1 = min(y + y_r + 1, height)
            errs[y0, x0] += 1
            errs[y0, x1] -= 1
            errs[y1, x0] -= 1
            errs[y1, x1] += 1

        # 2. Calculate cumulative sums
        errs = np.cumsum(errs, axis=0)
        errs = np.cumsum(errs, axis=1)

        # 3. Modify data
        locs = 5 * errs
        scales = 10 * np.sqrt(errs / 12) + 4 * errs
        for j in range(3):
            add = random_state.normal(locs, scales)
            if j == 2:
                add += 30 * errs
            if self.range == 1:
                node_data[:, :, j] = np.clip(node_data[:, :, j] + add[0:height, 0:width] / 255, 0, 1)
            else:
                node_data[:, :, j] = np.clip(node_data[:, :, j] + add[0:height, 0:width].astype(int), 0, 255)


class Snow(Filter):
    """Add snow to images.

    This filter adds snow to images, and it uses Pierrre Vigier's implementation
    of 2d perlin noise.

    Pierre Vigier's implementation of 2d perlin noise with slight changes.
    https://github.com/pvigier/perlin-numpy

    The original code is licensed under MIT License:

    MIT License

    Copyright (c) 2019 Pierre Vigier

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

    Inherits Filter class.
    """

    def __init__(self, snowflake_probability_id, snowflake_alpha_id, snowstorm_alpha_id):
        """
        Args:
            snowflake_probability_id (str): A key which maps to a snowflake probability.
            snowflake_alpha_id (str):
            snowstorm_alpha_id (str):
        """
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
            """[summary]

            Pierre Vigier's implementation of 2d perlin noise with slight changes.
            https://github.com/pvigier/perlin-numpy

            The original code is licensed under MIT License:

            MIT License

            Copyright (c) 2019 Pierre Vigier

            Permission is hereby granted, free of charge, to any person obtaining a copy
            of this software and associated documentation files (the "Software"), to deal
            in the Software without restriction, including without limitation the rights
            to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
            copies of the Software, and to permit persons to whom the Software is
            furnished to do so, subject to the following conditions:

            The above copyright notice and this permission notice shall be included in all
            copies or substantial portions of the Software.

            THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
            IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
            FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
            AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
            LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
            OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
            SOFTWARE.
            """

            def f(t):
                return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3

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
            n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
            n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
            return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)

        def build_snowflake(r):
            res = np.zeros(shape=(2 * r + 1, 2 * r + 1))
            for y in range(0, 2 * r + 1):
                for x in range(0, 2 * r + 1):
                    dy = y - r
                    dx = x - r
                    d = sqrt(dx * dx + dy * dy)
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
            y0 = max(0, y - r)
            x0 = max(0, x - r)
            y1 = min(height - 1, y + r) + 1
            x1 = min(width - 1, x + r) + 1
            fy0 = y0 - (y - r)
            fx0 = x0 - (x - r)
            fy1 = y1 - (y - r)
            fx1 = x1 - (x - r)
            for j in range(3):
                node_data[y0:y1, x0:x1, j] += ((255 - node_data[y0:y1, x0:x1, j]) * flakes[r][fy0:fy1, fx0:fx1]).astype(
                    node_data.dtype)

        # add noise
        noise = generate_perlin_noise(height, width, random_state)
        noise = (noise + 1) / 2  # transform the noise to be in range [0, 1]
        for j in range(3):
            node_data[:, :, j] += (self.snowstorm_alpha * (255 - node_data[:, :, j]) * noise[:, :]).astype(
                node_data.dtype)


class StainArea(Filter):
    """Adds stains to images.

    This filter adds stains to the images.
        probability: probability of adding a stain at each pixel.
        radius_generator: object implementing a generate(random_state) function
            which returns the radius of the stain.
        transparency_percentage: 1 means that the stain is invisible and 0 means
            that the part of the image where the stain is is completely black.

    Inherits Filter class.
    """

    def __init__(self, probability_id, radius_generator_id, transparency_percentage_id):
        """
        Args:
            probability_id (str): A key which maps to the probability of stain.
            radius_generator_id (str): A key which maps to the radius_generator.
            transparency_percentage_id (str): A key which maps to the transparency percentage.
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
        errs = np.zeros(shape=(height + 1, width + 1))
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


class Saturation(Filter):
    """Increases or decreases saturation in the image.

    tar: 0 if you want to decrease saturation, 1 if you want to increase it.

    rat: scales the saturation change.

    range: Should have value 1 or 255. The value is chosen according to how RGB values are presented in
    the corresponding NumPy array. Normally the values are either in the range [0,1] or in the set
    {0,...,255}. If this value is chosen incorrectly, then the filter will produce undesired
    effects on the image.

    Inherits Filter class.
    """

    def __init__(self, tar_id, rat_id, range_id):
        """
        Args:
            tar_id (str): A key which maps to the tar value.
            rat_id (str): A key which maps to the rat value.
            range_id (str): A key which maps to the range value.
        """
        super().__init__()
        self.tar_id = tar_id
        self.rat_id = rat_id
        self.range_id = range_id

    def set_params(self, params_dict):
        self.tar = params_dict[self.tar_id]
        self.rat = params_dict[self.rat_id]
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
            nd = nd.astype(np.uint8)
        else:
            nd = np.clip(nd, 0.0, 1.0)

        node_data[...] = nd


class LensFlare(Filter):
    """Add lens flare to an image.

    Inherits Filter class.
    """

    def __init__(self):
        super().__init__()

    def set_params(self, params_dict):
        pass

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
                    offset_dist = sqrt((x - x0 + x_offset) ** 2 + (y - y0 + y_offset) ** 2)
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
            if (best_x - width / 2) ** 2 + (best_y - height / 2) ** 2 + 1 <= (x - width / 2) ** 2 + (
                    y - height / 2) ** 2:
                break
            y += origo_vector[1]
            x += origo_vector[0]
            steps -= 1

# MIT License
#
# Copyright (c) 2019 Tuomas Halvari, Juha Harviainen, Juha Mylläri, Antti Röyskö, Juuso Silvennoinen
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

import numpy as np
from dpemu.nodes import Array
from dpemu import radius_generators
from dpemu.filters.image import Rain, Snow, StainArea, Blur, JPEG_Compression, BlurGaussian, Resolution, Rotation
from dpemu.filters.image import Brightness, Saturation


def test_seed_determines_result_for_fastrain_filter():
    a = np.zeros((10, 10, 3), dtype=int)
    x_node = Array()
    x_node.addfilter(Rain("probability", "range"))
    params = {"probability": 0.03, "range": 255}
    out1 = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    out2 = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.array_equal(out1, out2)


def test_seed_determines_result_for_fastrain_filter_two():
    a = np.zeros((10, 10, 3), dtype=int)
    x_node = Array()
    x_node.addfilter(Rain("probability", "range"))
    params = {"probability": 0.03, "range": 1}
    out1 = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    out2 = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.array_equal(out1, out2)


def test_seed_determines_result_for_snow_filter():
    a = np.zeros((10, 10, 3), dtype=int)
    x_node = Array()
    x_node.addfilter(Snow("snowflake_probability", "snowflake_alpha", "snowstorm_alpha"))
    params = {"snowflake_probability": 0.04, "snowflake_alpha": 0.4, "snowstorm_alpha": 1}
    out1 = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    out2 = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.array_equal(out1, out2)


def test_seed_determines_result_for_stain_filter():
    def f(data, random_state):
        return data * random_state.randint(2, 4)

    a = np.random.RandomState(seed=42).randint(0, 255, size=300).reshape((10, 10, 3))
    x_node = Array()
    x_node.addfilter(StainArea("probability", "radius_generator", "transparency_percentage"))
    params = {"probability": .005,
              "radius_generator": radius_generators.GaussianRadiusGenerator(10, 5),
              "transparency_percentage": 0.5}
    out1 = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    out2 = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.array_equal(out1, out2)


def test_blur_iterates_correctly():
    rs = np.random.RandomState(seed=42)
    dat1 = rs.randint(low=0, high=255, size=(10, 10, 3))
    dat2 = dat1.copy()

    blur_once = Blur("repeats")
    blur_once.set_params({"repeats": 1})
    blur_once.apply(dat1, rs, named_dims={})
    blur_once.apply(dat1, rs, named_dims={})

    blur_twice = Blur("repeats")
    blur_twice.set_params({"repeats": 2})
    blur_twice.apply(dat2, rs, named_dims={})

    assert np.array_equal(dat1, dat2)


def test_gaussian_blur_works_for_different_shapes():
    rs = np.random.RandomState(seed=42)
    dat1 = rs.randint(low=0, high=255, size=(10, 10, 3))
    dat2 = dat1.copy()

    blur = BlurGaussian("std")
    blur.set_params({"std": 5})
    blur.apply(dat1, rs, named_dims={})
    for j in range(3):
        blur.apply(dat2[:, :, j], rs, named_dims={})
    assert np.array_equal(dat1, dat2)


def test_resolution():
    scale = 3
    height = 50
    width = 50
    rs = np.random.RandomState(seed=42)
    dat1 = rs.randint(low=0, high=255, size=(height, width, 3))
    dat2 = dat1.copy()

    for y in range(height):
        ry = (y // scale) * scale
        for x in range(width):
            rx = (x // scale) * scale
            dat1[y, x, :] = dat1[ry, rx, :]

    res = Resolution("scale")
    res.set_params({"scale": scale})
    res.apply(dat2, rs, named_dims={})

    assert np.array_equal(dat1, dat2)


def test_jpeg_compression():
    rs = np.random.RandomState(seed=42)
    dat = rs.randint(low=0, high=255, size=(50, 50))
    orig_dat = np.uint8(dat)

    comp = JPEG_Compression("quality")
    params = {"quality": 50}
    comp.set_params(params)
    comp.apply(dat, rs, named_dims={})
    dat = np.uint8(dat)

    assert not (abs(dat - orig_dat) < 5).all()


def test_rotation_creates_no_black_pixels_on_wide_images():
    shape = (71, 217, 3)
    data = np.zeros(shape) + 255

    for angle in range(360):
        rot = Rotation("angle", "angle")
        rot.set_params({"angle": angle})
        rot.apply(data, np.random.RandomState(42), named_dims={})
    assert np.min(data) >= 254.9


def test_rotation_creates_no_black_pixels_on_high_images():
    shape = (217, 71, 3)
    data = np.zeros(shape) + 255

    for angle in range(360):
        rot = Rotation("angle", "angle")
        rot.set_params({"angle": angle})
        rot.apply(data, np.random.RandomState(42), named_dims={})
    assert np.min(data) >= 254.9


def test_brightness_brightens_image():
    rs = np.random.RandomState(seed=42)
    shape = (100, 100, 3)
    data = rs.randint(low=0, high=255, size=shape)
    original = data.copy()
    ftr = Brightness("tar", "rat", "range")
    ftr.set_params({"tar": 1, "rat": 0.5, "range": 255})
    ftr.apply(data, rs, named_dims={})
    assert np.sum(data.astype(int) - original.astype(int)) > 0 and np.min(data - original) >= 0


def test_saturation_saturates_image():
    rs = np.random.RandomState(seed=42)
    shape = (100, 100, 3)
    data = rs.randint(low=0, high=255, size=shape)
    original = data.copy()
    ftr = Saturation("tar", "rat", "range")
    ftr.set_params({"tar": 1, "rat": 0.5, "range": 255})
    ftr.apply(data, rs, named_dims={})

    original_diff = np.maximum(np.maximum(original[:, :, 0], original[:, :, 1]), original[:, :, 2])
    original_diff -= np.minimum(np.minimum(original[:, :, 0], original[:, :, 1]), original[:, :, 2])

    diff = np.maximum(np.maximum(data[:, :, 0], data[:, :, 1]), data[:, :, 2])
    diff -= np.minimum(np.minimum(data[:, :, 0], data[:, :, 1]), data[:, :, 2])

    assert np.sum(diff.astype(int) - original_diff.astype(int)) > 0 and np.min(diff - original_diff) >= 0

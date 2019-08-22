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
from dpemu.filters.text import Uppercase, OCRError, MissingArea


def test_seed_determines_result_for_uppercase_filter():
    a = np.array(["hello world"])
    x_node = Array()
    x_node.addfilter(Uppercase("prob"))
    params = {"prob": .5}
    out1 = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    out2 = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.alltrue(out1 == out2)


def test_seed_determines_result_for_ocr_error_filter():
    a = np.array(["hello world"])
    x_node = Array()
    x_node.addfilter(OCRError("probs", "p"))
    params = {"probs": {"e": (["E", "i"], [.5, .5]), "g": (["q", "9"], [.2, .8])}, "p": 1}
    out1 = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    out2 = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.array_equal(out1, out2)


def test_seed_determines_result_for_missing_area_filter_with_gaussian_radius_generator():
    a = np.array(["hello world\n" * 10])
    x_node = Array()
    x_node.addfilter(MissingArea("probability", "radius_generator", "missing_value"))
    params = {"probability": 0.05,
              "radius_generator": radius_generators.GaussianRadiusGenerator(1, 1),
              "missing_value": "#"}
    out1 = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    out2 = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.array_equal(out1, out2)


def test_seed_determines_result_for_missing_area_filter_with_probability_array_radius_generator():
    a = np.array(["hello world\n" * 10])
    x_node = Array()
    x_node.addfilter(MissingArea("probability", "radius_generator", "missing_value"))
    params = {"probability": 0.05,
              "radius_generator": radius_generators.ProbabilityArrayRadiusGenerator([.6, .3, .1]),
              "missing_value": "#"}
    out1 = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    out2 = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.array_equal(out1, out2)

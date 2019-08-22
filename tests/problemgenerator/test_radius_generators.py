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

from dpemu import radius_generators


def test_probability_array_radius_generator_returns_zero_if_no_radius_is_chosen():
    r = radius_generators.ProbabilityArrayRadiusGenerator([0, 0, 0])
    assert r.generate(np.random.RandomState(seed=42)) == 0


def test_probability_array_radius_generator_returns_distinct_values():
    r = radius_generators.ProbabilityArrayRadiusGenerator([.3, .4, .3])
    rs = np.random.RandomState(seed=42)
    s = set()
    for _ in range(0, 50):
        s.add(r.generate(rs))
    assert len(s) != 1


def test_gaussian_radius_generator_returns_distinct_values():
    r = radius_generators.GaussianRadiusGenerator(3, 2)
    rs = np.random.RandomState(seed=42)
    s = set()
    for _ in range(0, 50):
        s.add(r.generate(rs))
    assert len(s) != 1

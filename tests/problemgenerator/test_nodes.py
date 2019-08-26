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

from dpemu.nodes import Array, Series, TupleSeries, Tuple
from dpemu.filters.common import Missing
from dpemu.filters.time_series import SensorDrift


def test_array_works_with_regular_arrays():
    a = [0.]
    x_node = Array()
    x_node.addfilter(Missing("prob", "m_val"))
    params = {"prob": 1., "m_val": np.nan}
    out = x_node.generate_error(a, params)
    assert np.isnan(out[0])


def test_series_and_array_work_with_regular_arrays():
    a = [0.]
    x_node = Array()
    x_node.addfilter(Missing("prob", "m_val"))
    series_node = Series(x_node)
    params = {"prob": 1., "m_val": np.nan}
    out = series_node.generate_error(a, params)
    assert np.isnan(out[0])


def test_array_works_with_numpy_arrays():
    a = np.array([0.])
    x_node = Array()
    x_node.addfilter(Missing("prob", "m_val"))
    params = {"prob": 1., "m_val": np.nan}
    out = x_node.generate_error(a, params)
    assert np.isnan(out[0])


def test_tuple_series_works_with_numpy_arrays():
    a = np.array([[1, 2, 3]])
    b = np.array([[1, 1, 1]])
    data = (a, b)
    x_node = Array()
    x_node.addfilter(SensorDrift("a"))
    y_node = Array()
    root_node = TupleSeries([x_node, y_node])
    res = root_node.generate_error(data, {'a': 1})
    assert np.array_equal(res[0], np.array([[2, 4, 6]])) and np.array_equal(res[1], np.array([[1, 1, 1]]))


def test_tuple_node_works():
    data = np.array([(0., 1.), (2., 3.)])
    x_node = Tuple()
    x_node.addfilter(Missing("prob", "m_val"))
    series_node = Series(x_node)
    data = series_node.generate_error(data, {'prob': 1, 'm_val': np.nan})
    assert np.isnan(data[0][0]) and np.isnan(data[0][1]) and np.isnan(data[1][0]) and np.isnan(data[1][1])


def test_exception_raised_when_param_missing():
    data = np.random.rand(5)
    x_node = Array()
    x_node.addfilter(Missing("prob", "m_val"))
    try:
        x_node.generate_error(data, {"probb": .5, "m_val": np.nan})
    except Exception as e:
        assert "prob" in str(e)

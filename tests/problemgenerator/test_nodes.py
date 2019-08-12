import numpy as np

from dpemu.nodes import Array, Series, TupleSeries
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

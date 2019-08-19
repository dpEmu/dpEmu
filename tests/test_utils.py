import numpy as np
from dpemu import utils


def test_expand_parameter_to_linspace():
    param1 = [2.0]
    param2 = [1.0, 50.0]
    assert np.array_equal(utils.expand_parameter_to_linspace(param1), np.array([2.0]))
    assert np.array_equal(utils.expand_parameter_to_linspace(param2), np.array([i + 1 for i in range(50)]))

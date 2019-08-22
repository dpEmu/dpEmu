import numpy as np
from dpemu.nodes import Array, Series
from dpemu.filters import Identity
from dpemu.filters.common import Missing, GaussianNoise, StrangeBehaviour, GaussianNoiseTimeDependent, Clip
from dpemu.filters.common import ModifyAsDataType, ApplyWithProbability
from dpemu.filters.text import OCRError


def test_apply_with_probability():
    data = np.array([["a"], ["a"], ["a"], ["a"], ["a"], ["a"], ["a"], ["a"], ["a"], ["a"]])

    ocr = OCRError("ps", "p")
    x_node = Array()
    x_node.addfilter(ApplyWithProbability(ocr, "ocr_prob"))
    series_node = Series(x_node)
    params = {"ps": {"a": [["e"], [1.0]]}, "p": 1.0, "ocr_prob": 0.5}
    out = series_node.generate_error(data, params, np.random.RandomState(seed=42))

    contains_distinct_elements = False
    for a in out:
        for b in out:
            if a != b:
                contains_distinct_elements = True
    assert contains_distinct_elements


def test_seed_determines_result_for_missing_filter():
    a = np.array([0., 1., 2., 3., 4.])
    x_node = Array()
    x_node.addfilter(Missing("prob", "m_val"))
    params = {"prob": .5, "m_val": np.nan}
    out1 = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    out2 = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.allclose(out1, out2, equal_nan=True)


def test_seed_determines_result_for_gaussian_noise_filter():
    a = np.array([0., 1., 2., 3., 4.])
    x_node = Array()
    x_node.addfilter(GaussianNoise("mean", "std"))
    params = {"mean": .5, "std": .5}
    out1 = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    out2 = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.allclose(out1, out2, equal_nan=True)


def test_seed_determines_result_for_strange_behaviour_filter():
    def f(data, random_state):
        return data * random_state.randint(2, 4)

    a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    x_node = Array()
    x_node.addfilter(StrangeBehaviour("f"))
    params = {"f": f}
    out1 = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    out2 = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.array_equal(out1, out2)


def test_seed_determines_result_for_time_dependent_gaussian_noise():
    a = np.arange(25).reshape((5, 5)).astype(np.float64)
    params = {}
    params['mean'] = 2.
    params['std'] = 3.
    params['mean_inc'] = 1.
    params['std_inc'] = 4.
    x_node = Array()
    x_node.addfilter(GaussianNoiseTimeDependent('mean', 'std', 'mean_inc', 'std_inc'))
    series_node = Series(x_node, dim_name="time")
    out1 = series_node.generate_error(a, params, np.random.RandomState(seed=42))
    out2 = series_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.allclose(out1, out2)


def test_strange_behaviour():
    def strange(x, _):
        if 15 <= x <= 20:
            return -300

        return x

    weird = StrangeBehaviour("strange")
    params = {"strange": strange}
    weird.set_params(params)
    y = np.arange(0, 30)
    weird.apply(y, np.random.RandomState(), named_dims={})

    for i in range(15, 21):
        assert y[i] == -300


def test_clip():
    a = np.arange(5)
    params = {}
    params['min'] = 2
    params['max'] = 3
    x_node = Array()
    x_node.addfilter(Clip('min', 'max'))
    out = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.array_equal(out, np.array([2, 2, 2, 3, 3]))


def test_modify_as_datatype():
    a = np.array([256 + 42])
    params = {}
    params['dtype'] = np.int8
    x_node = Array()
    x_node.addfilter(ModifyAsDataType('dtype', Identity()))
    out = x_node.generate_error(a, params)
    assert np.array_equal(out, np.array([42]))

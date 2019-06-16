import numpy as np

import src.problemgenerator.array as array
import src.problemgenerator.copy as copy
import src.problemgenerator.filters as filters
import src.problemgenerator.radius_generators as radius_generators


def test_seed_determines_result_for_missing_filter():
    a = np.array([0., 1., 2., 3., 4.])
    x_node = array.Array(a.shape)
    x_node.addfilter(filters.Missing(0.5))
    root_node = copy.Copy(x_node)
    out1 = root_node.process(a, np.random.RandomState(seed=42))
    out2 = root_node.process(a, np.random.RandomState(seed=42))
    assert np.allclose(out1, out2, equal_nan=True)


def test_seed_determines_result_for_gaussian_noise_filter():
    a = np.array([0., 1., 2., 3., 4.])
    x_node = array.Array(a.shape)
    x_node.addfilter(filters.GaussianNoise(0.5, 0.5))
    root_node = copy.Copy(x_node)
    out1 = root_node.process(a, np.random.RandomState(seed=42))
    out2 = root_node.process(a, np.random.RandomState(seed=42))
    assert np.allclose(out1, out2, equal_nan=True)


def test_seed_determines_result_for_uppercase_filter():
    a = np.array(["hello world"])
    x_node = array.Array(a.shape)
    x_node.addfilter(filters.Uppercase(0.5))
    root_node = copy.Copy(x_node)
    out1 = root_node.process(a, np.random.RandomState(seed=42))
    out2 = root_node.process(a, np.random.RandomState(seed=42))
    assert np.array_equal(out1, out2)


def test_seed_determines_result_for_ocr_error_filter():
    a = np.array(["hello world"])
    x_node = array.Array(a.shape)
    x_node.addfilter(filters.OCRError({"e": (["E", "i"], [.5, .5]), "g": (["q", "9"], [.2, .8])}, 1))
    root_node = copy.Copy(x_node)
    out1 = root_node.process(a, np.random.RandomState(seed=42))
    out2 = root_node.process(a, np.random.RandomState(seed=42))
    assert np.array_equal(out1, out2)


def test_seed_determines_result_for_missing_area_filter_with_gaussian_radius_generator():
    a = np.array(["hello world\n" * 10])
    x_node = array.Array(a.shape)
    x_node.addfilter(filters.MissingArea(0.05, radius_generators.GaussianRadiusGenerator(1, 1), "#"))
    root_node = copy.Copy(x_node)
    out1 = root_node.process(a, np.random.RandomState(seed=42))
    out2 = root_node.process(a, np.random.RandomState(seed=42))
    assert np.array_equal(out1, out2)


def test_seed_determines_result_for_missing_area_filter_with_probability_array_radius_generator():
    a = np.array(["hello world\n" * 10])
    x_node = array.Array(a.shape)
    x_node.addfilter(filters.MissingArea(0.05, radius_generators.ProbabilityArrayRadiusGenerator([.6, .3, .1]), "#"))
    root_node = copy.Copy(x_node)
    out1 = root_node.process(a, np.random.RandomState(seed=42))
    out2 = root_node.process(a, np.random.RandomState(seed=42))
    assert np.array_equal(out1, out2)


def test_seed_determines_result_for_gap_filter():
    a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    x_node = array.Array(a.shape)
    x_node.addfilter(filters.Gap(0.1, 0.1, missing_value=1337))
    root_node = copy.Copy(x_node)
    out1 = root_node.process(a, np.random.RandomState(seed=42))
    out2 = root_node.process(a, np.random.RandomState(seed=42))
    assert np.array_equal(out1, out2)


def test_seed_determines_result_for_strange_behaviour_filter():
    def f(data, random_state):
        return data * random_state.randint(2, 4)

    a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    x_node = array.Array(a.shape)
    x_node.addfilter(filters.StrangeBehaviour(f))
    root_node = copy.Copy(x_node)
    out1 = root_node.process(a, np.random.RandomState(seed=42))
    out2 = root_node.process(a, np.random.RandomState(seed=42))
    assert np.array_equal(out1, out2)


def test_seed_determines_result_for_rain_filter():
    a = np.zeros((10, 10, 3), dtype=int)
    x_node = array.Array(a.shape)
    x_node.addfilter(filters.Rain(0.03))
    root_node = copy.Copy(x_node)
    out1 = root_node.process(a, np.random.RandomState(seed=42))
    out2 = root_node.process(a, np.random.RandomState(seed=42))
    assert np.array_equal(out1, out2)


def test_seed_determines_result_for_snow_filter():
    a = np.zeros((10, 10, 3), dtype=int)
    x_node = array.Array(a.shape)
    x_node.addfilter(filters.Snow(0.04, 0.4, 1))
    root_node = copy.Copy(x_node)
    out1 = root_node.process(a, np.random.RandomState(seed=42))
    out2 = root_node.process(a, np.random.RandomState(seed=42))
    assert np.array_equal(out1, out2)


def test_seed_determines_result_for_blur_filter():
    def f(data, random_state):
        return data * random_state.randint(2, 4)

    a = np.random.RandomState(seed=42).randint(0, 255, size=300).reshape((10, 10, 3))
    x_node = array.Array(a.shape)
    x_node.addfilter(filters.Blur(5))
    root_node = copy.Copy(x_node)
    out1 = root_node.process(a, np.random.RandomState(seed=42))
    out2 = root_node.process(a, np.random.RandomState(seed=42))
    assert np.array_equal(out1, out2)

def test_seed_determines_result_for_stain_filter():
    def f(data, random_state):
        return data * random_state.randint(2, 4)

    a = np.random.RandomState(seed=42).randint(0, 255, size=300).reshape((10, 10, 3))
    x_node = array.Array(a.shape)
    x_node.addfilter(filters.StainArea(.002, radius_generators.GaussianRadiusGenerator(10, 5), 0.5))
    root_node = copy.Copy(x_node)
    out1 = root_node.process(a, np.random.RandomState(seed=42))
    out2 = root_node.process(a, np.random.RandomState(seed=42))
    assert np.array_equal(out1, out2)


def test_sensor_drift():
    drift = filters.SensorDrift(1)
    y = np.full((100), 1)
    drift.apply(y, np.random.RandomState(), (), named_dims={})

    increases = np.arange(1, 101)

    assert len(y) == len(increases)
    for i, val in enumerate(y):
        assert val == increases[i] + 1


def test_strange_behaviour():
    def strange(x, _):
        if 15 <= x <= 20:
            return -300

        return x

    weird = filters.StrangeBehaviour(strange)
    y = np.arange(0, 30)
    weird.apply(y, np.random.RandomState(), (), named_dims={})

    for i in range(15, 21):
        assert y[i] == -300


def test_one_gap():
    gap = filters.Gap(0.0, 1)
    y = np.arange(10000.0)
    gap.apply(y, np.random.RandomState(), (), named_dims={})

    for _, val in enumerate(y):
        assert not np.isnan(val)


def test_two_gap():
    gap = filters.Gap(1, 0)
    y = np.arange(10000.0)
    gap.apply(y, np.random.RandomState(), (), named_dims={})

    for _, val in enumerate(y):
        assert np.isnan(val)

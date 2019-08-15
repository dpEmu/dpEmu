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

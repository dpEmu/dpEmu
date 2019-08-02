import numpy as np

import src.problemgenerator.radius_generators as radius_generators


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

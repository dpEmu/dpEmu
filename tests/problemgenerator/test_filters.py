import numpy as np

from dpemu.nodes import Array
from dpemu.nodes import Series
from dpemu.problemgenerator import filters
from dpemu import radius_generators


def test_seed_determines_result_for_missing_filter():
    a = np.array([0., 1., 2., 3., 4.])
    x_node = Array()
    x_node.addfilter(filters.Missing("prob"))
    params = {"prob": .5}
    out1 = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    out2 = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.allclose(out1, out2, equal_nan=True)


def test_seed_determines_result_for_gaussian_noise_filter():
    a = np.array([0., 1., 2., 3., 4.])
    x_node = Array()
    x_node.addfilter(filters.GaussianNoise("mean", "std"))
    params = {"mean": .5, "std": .5}
    out1 = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    out2 = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.allclose(out1, out2, equal_nan=True)


def test_seed_determines_result_for_uppercase_filter():
    a = np.array(["hello world"])
    x_node = Array()
    x_node.addfilter(filters.Uppercase("prob"))
    params = {"prob": .5}
    out1 = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    out2 = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.alltrue(out1 == out2)


def test_seed_determines_result_for_ocr_error_filter():
    a = np.array(["hello world"])
    x_node = Array()
    x_node.addfilter(filters.OCRError("probs", "p"))
    params = {"probs": {"e": (["E", "i"], [.5, .5]), "g": (["q", "9"], [.2, .8])}, "p": 1}
    out1 = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    out2 = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.array_equal(out1, out2)


def test_seed_determines_result_for_missing_area_filter_with_gaussian_radius_generator():
    a = np.array(["hello world\n" * 10])
    x_node = Array()
    x_node.addfilter(filters.MissingArea("probability", "radius_generator", "missing_value"))
    params = {"probability": 0.05,
              "radius_generator": radius_generators.GaussianRadiusGenerator(1, 1),
              "missing_value": "#"}
    out1 = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    out2 = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.array_equal(out1, out2)


def test_seed_determines_result_for_missing_area_filter_with_probability_array_radius_generator():
    a = np.array(["hello world\n" * 10])
    x_node = Array()
    x_node.addfilter(filters.MissingArea("probability", "radius_generator", "missing_value"))
    params = {"probability": 0.05,
              "radius_generator": radius_generators.ProbabilityArrayRadiusGenerator([.6, .3, .1]),
              "missing_value": "#"}
    out1 = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    out2 = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.array_equal(out1, out2)


def test_seed_determines_result_for_gap_filter():
    a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    x_node = Array()
    x_node.addfilter(filters.Gap("prob_break", "prob_recover", "missing_value"))
    params = {"prob_break": 0.1, "prob_recover": 0.1, "missing_value": 1337}
    out1 = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    out2 = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.array_equal(out1, out2)


def test_seed_determines_result_for_strange_behaviour_filter():
    def f(data, random_state):
        return data * random_state.randint(2, 4)

    a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    x_node = Array()
    x_node.addfilter(filters.StrangeBehaviour("f"))
    params = {"f": f}
    out1 = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    out2 = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.array_equal(out1, out2)


def test_seed_determines_result_for_fastrain_filter():
    a = np.zeros((10, 10, 3), dtype=int)
    x_node = Array()
    x_node.addfilter(filters.FastRain("probability", "range"))
    params = {"probability": 0.03, "range": 255}
    out1 = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    out2 = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.array_equal(out1, out2)


def test_seed_determines_result_for_fastrain_filter_two():
    a = np.zeros((10, 10, 3), dtype=int)
    x_node = Array()
    x_node.addfilter(filters.FastRain("probability", "range"))
    params = {"probability": 0.03, "range": 1}
    out1 = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    out2 = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.array_equal(out1, out2)


def test_seed_determines_result_for_snow_filter():
    a = np.zeros((10, 10, 3), dtype=int)
    x_node = Array()
    x_node.addfilter(filters.Snow("snowflake_probability", "snowflake_alpha", "snowstorm_alpha"))
    params = {"snowflake_probability": 0.04, "snowflake_alpha": 0.4, "snowstorm_alpha": 1}
    out1 = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    out2 = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.array_equal(out1, out2)


def test_seed_determines_result_for_stain_filter():
    def f(data, random_state):
        return data * random_state.randint(2, 4)

    a = np.random.RandomState(seed=42).randint(0, 255, size=300).reshape((10, 10, 3))
    x_node = Array()
    x_node.addfilter(filters.StainArea("probability", "radius_generator", "transparency_percentage"))
    params = {"probability": .005,
              "radius_generator": radius_generators.GaussianRadiusGenerator(10, 5),
              "transparency_percentage": 0.5}
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
    x_node.addfilter(filters.GaussianNoiseTimeDependent('mean', 'std', 'mean_inc', 'std_inc'))
    series_node = Series(x_node, dim_name="time")
    out1 = series_node.generate_error(a, params, np.random.RandomState(seed=42))
    out2 = series_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.allclose(out1, out2)


def test_blur_iterates_correctly():
    rs = np.random.RandomState(seed=42)
    dat1 = rs.randint(low=0, high=255, size=(10, 10, 3))
    dat2 = dat1.copy()

    blur_once = filters.Blur("repeats")
    blur_once.set_params({"repeats": 1})
    blur_once.apply(dat1, rs, named_dims={})
    blur_once.apply(dat1, rs, named_dims={})

    blur_twice = filters.Blur("repeats")
    blur_twice.set_params({"repeats": 2})
    blur_twice.apply(dat2, rs, named_dims={})

    assert np.array_equal(dat1, dat2)


def test_jpeg_compression():
    rs = np.random.RandomState(seed=42)
    dat = rs.randint(low=0, high=255, size=(50, 50))
    orig_dat = np.uint8(dat)

    comp = filters.JPEG_Compression("quality")
    params = {"quality": 50}
    comp.set_params(params)
    comp.apply(dat, rs, named_dims={})
    dat = np.uint8(dat)

    assert not (abs(dat - orig_dat) < 5).all()


def test_sensor_drift():
    drift = filters.SensorDrift("magnitude")
    params = {"magnitude": 2}
    drift.set_params(params)
    y = np.full((100), 1)
    drift.apply(y, np.random.RandomState(), named_dims={})

    increases = np.arange(1, 101)

    assert len(y) == len(increases)
    for i, val in enumerate(y):
        assert val == increases[i]*2 + 1


def test_strange_behaviour():
    def strange(x, _):
        if 15 <= x <= 20:
            return -300

        return x

    weird = filters.StrangeBehaviour("strange")
    params = {"strange": strange}
    weird.set_params(params)
    y = np.arange(0, 30)
    weird.apply(y, np.random.RandomState(), named_dims={})

    for i in range(15, 21):
        assert y[i] == -300


def test_one_gap():
    gap = filters.Gap("prob_break", "prob_recover", "missing")
    y = np.arange(10000.0)
    params = {"prob_break": 0.0, "prob_recover": 1, "missing": np.nan}
    gap.set_params(params)
    gap.apply(y, np.random.RandomState(), named_dims={})

    for _, val in enumerate(y):
        assert not np.isnan(val)


def test_two_gap():
    gap = filters.Gap("prob_break", "prob_recover", "missing")
    params = {"prob_break": 1.0, "prob_recover": 0.0, "missing": np.nan}
    y = np.arange(10000.0)
    gap.set_params(params)
    gap.apply(y, np.random.RandomState(), named_dims={})

    for _, val in enumerate(y):
        assert np.isnan(val)


def test_apply_with_probability():
    data = np.array([["a"], ["a"], ["a"], ["a"], ["a"], ["a"], ["a"], ["a"], ["a"], ["a"]])

    ocr = filters.OCRError("ps", "p")
    x_node = Array()
    x_node.addfilter(filters.ApplyWithProbability("ocr_node", "ocr_prob"))
    series_node = Series(x_node)
    params = {"ps": {"a": [["e"], [1.0]]}, "p": 1.0, "ocr_node": ocr, "ocr_prob": 0.5}
    out = series_node.generate_error(data, params, np.random.RandomState(seed=42))

    contains_distinct_elements = False
    for a in out:
        for b in out:
            if a != b:
                contains_distinct_elements = True
    assert contains_distinct_elements


def test_constant():
    a = np.arange(25).reshape((5, 5))
    params = {'c': 5}
    x_node = Array()
    x_node.addfilter(filters.Constant('c'))
    out = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.array_equal(out, np.full((5, 5), 5))


def test_identity():
    a = np.arange(25).reshape((5, 5))
    x_node = Array()
    x_node.addfilter(filters.Identity())
    out = x_node.generate_error(a, {}, np.random.RandomState(seed=42))
    assert np.array_equal(out, a)


def test_addition():
    a = np.full((5, 5), 5)
    params = {}
    params['const'] = filters.Constant('c')
    params['c'] = 2
    params['identity'] = filters.Identity()
    x_node = Array()
    x_node.addfilter(filters.Addition('const', 'identity'))
    out = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.array_equal(out, np.full((5, 5), 7))


def test_subtraction():
    a = np.full((5, 5), 5)
    params = {}
    params['const'] = filters.Constant('c')
    params['c'] = 2
    params['identity'] = filters.Identity()
    x_node = Array()
    x_node.addfilter(filters.Subtraction('const', 'identity'))
    out = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.array_equal(out, np.full((5, 5), -3))


def test_multiplication():
    a = np.full((5, 5), 5)
    params = {}
    params['const'] = filters.Constant('c')
    params['c'] = 2
    params['identity'] = filters.Identity()
    x_node = Array()
    x_node.addfilter(filters.Multiplication('const', 'identity'))
    out = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.array_equal(out, np.full((5, 5), 10))


def test_division():
    a = np.full((5, 5), 5.0)
    params = {}
    params['const'] = filters.Constant('c')
    params['c'] = 2
    params['identity'] = filters.Identity()
    x_node = Array()
    x_node.addfilter(filters.Division('const', 'identity'))
    out = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.allclose(out, np.full((5, 5), .4))


def test_integer_division():
    a = np.full((5, 5), 5)
    params = {}
    params['const'] = filters.Constant('c')
    params['c'] = 2
    params['identity'] = filters.Identity()
    x_node = Array()
    x_node.addfilter(filters.IntegerDivision('identity', 'const'))
    out = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.array_equal(out, np.full((5, 5), 2))


def test_modulo():
    a = np.full((5, 5), 5)
    params = {}
    params['const'] = filters.Constant('c')
    params['c'] = 2
    params['identity'] = filters.Identity()
    x_node = Array()
    x_node.addfilter(filters.Modulo('identity', 'const'))
    out = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.array_equal(out, np.full((5, 5), 1))


def test_and():
    a = np.full((5, 5), 5)
    params = {}
    params['const'] = filters.Constant('c')
    params['c'] = 2
    params['identity'] = filters.Identity()
    x_node = Array()
    x_node.addfilter(filters.And('identity', 'const'))
    out = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.array_equal(out, np.full((5, 5), 0))


def test_or():
    a = np.full((5, 5), 5)
    params = {}
    params['const'] = filters.Constant('c')
    params['c'] = 2
    params['identity'] = filters.Identity()
    x_node = Array()
    x_node.addfilter(filters.Or('identity', 'const'))
    out = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.array_equal(out, np.full((5, 5), 7))


def test_xor():
    a = np.full((5, 5), 5)
    params = {}
    params['const'] = filters.Constant('c')
    params['c'] = 3
    params['identity'] = filters.Identity()
    x_node = Array()
    x_node.addfilter(filters.Xor('identity', 'const'))
    out = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.array_equal(out, np.full((5, 5), 6))


def test_difference():
    a = np.full((5, 5), 5)
    params = {}
    params['const'] = filters.Constant('c')
    params['c'] = 2
    params['identity'] = filters.Identity()
    params['addition'] = filters.Addition('identity', 'const')
    x_node = Array()
    x_node.addfilter(filters.Difference("addition"))
    out = x_node.generate_error(a, params, np.random.RandomState(seed=42))

    assert np.array_equal(out, np.full((5, 5), 2))


def test_min():
    a = np.full((5, 5), 5)
    params = {}
    params['const'] = filters.Constant('c')
    params['c'] = 2
    params['identity'] = filters.Identity()
    x_node = Array()
    x_node.addfilter(filters.Min('identity', 'const'))
    out = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.array_equal(out, np.full((5, 5), 2))


def test_max():
    a = np.full((5, 5), 5)
    params = {}
    params['const'] = filters.Constant('c')
    params['c'] = 2
    params['identity'] = filters.Identity()
    x_node = Array()
    x_node.addfilter(filters.Max('identity', 'const'))
    out = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.array_equal(out, np.full((5, 5), 5))


def test_clip():
    a = np.arange(5)
    params = {}
    params['min'] = 2
    params['max'] = 3
    x_node = Array()
    x_node.addfilter(filters.Clip('min', 'max'))
    out = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.array_equal(out, np.array([2, 2, 2, 3, 3]))


def test_modify_as_datatype():
    a = np.array([256 + 42])
    params = {}
    params['dtype'] = np.int8
    params['filter'] = filters.Identity()
    x_node = Array()
    x_node.addfilter(filters.ModifyAsDataType('dtype', 'filter'))
    out = x_node.generate_error(a, params)
    assert np.array_equal(out, np.array([42]))

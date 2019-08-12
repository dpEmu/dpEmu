import numpy as np

from dpemu.nodes import Array
from dpemu.nodes import Series
from dpemu.filters import Constant, Addition, Subtraction, Multiplication, Division, IntegerDivision, Identity
from dpemu.filters import Min, Max, Difference, Modulo, And, Or, Xor
from dpemu.filters.common import Missing, GaussianNoise, StrangeBehaviour, GaussianNoiseTimeDependent, Clip
from dpemu.filters.common import ModifyAsDataType, ApplyWithProbability
from dpemu.filters.image import Rain, Snow, StainArea, Blur, JPEG_Compression
from dpemu.filters.text import Uppercase, OCRError, MissingArea
from dpemu.filters.time_series import Gap, SensorDrift
from dpemu import radius_generators


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


def test_seed_determines_result_for_gap_filter():
    a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    x_node = Array()
    x_node.addfilter(Gap("prob_break", "prob_recover", "missing_value"))
    params = {"prob_break": 0.1, "prob_recover": 0.1, "missing_value": 1337}
    out1 = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    out2 = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.array_equal(out1, out2)


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


def test_seed_determines_result_for_fastrain_filter():
    a = np.zeros((10, 10, 3), dtype=int)
    x_node = Array()
    x_node.addfilter(Rain("probability", "range"))
    params = {"probability": 0.03, "range": 255}
    out1 = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    out2 = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.array_equal(out1, out2)


def test_seed_determines_result_for_fastrain_filter_two():
    a = np.zeros((10, 10, 3), dtype=int)
    x_node = Array()
    x_node.addfilter(Rain("probability", "range"))
    params = {"probability": 0.03, "range": 1}
    out1 = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    out2 = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.array_equal(out1, out2)


def test_seed_determines_result_for_snow_filter():
    a = np.zeros((10, 10, 3), dtype=int)
    x_node = Array()
    x_node.addfilter(Snow("snowflake_probability", "snowflake_alpha", "snowstorm_alpha"))
    params = {"snowflake_probability": 0.04, "snowflake_alpha": 0.4, "snowstorm_alpha": 1}
    out1 = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    out2 = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.array_equal(out1, out2)


def test_seed_determines_result_for_stain_filter():
    def f(data, random_state):
        return data * random_state.randint(2, 4)

    a = np.random.RandomState(seed=42).randint(0, 255, size=300).reshape((10, 10, 3))
    x_node = Array()
    x_node.addfilter(StainArea("probability", "radius_generator", "transparency_percentage"))
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
    x_node.addfilter(GaussianNoiseTimeDependent('mean', 'std', 'mean_inc', 'std_inc'))
    series_node = Series(x_node, dim_name="time")
    out1 = series_node.generate_error(a, params, np.random.RandomState(seed=42))
    out2 = series_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.allclose(out1, out2)


def test_blur_iterates_correctly():
    rs = np.random.RandomState(seed=42)
    dat1 = rs.randint(low=0, high=255, size=(10, 10, 3))
    dat2 = dat1.copy()

    blur_once = Blur("repeats")
    blur_once.set_params({"repeats": 1})
    blur_once.apply(dat1, rs, named_dims={})
    blur_once.apply(dat1, rs, named_dims={})

    blur_twice = Blur("repeats")
    blur_twice.set_params({"repeats": 2})
    blur_twice.apply(dat2, rs, named_dims={})

    assert np.array_equal(dat1, dat2)


def test_gaussian_blur_works_for_different_shapes():
    rs = np.random.RandomState(seed=42)
    dat1 = rs.randint(low=0, high=255, size=(10, 10, 3))
    dat2 = dat1.copy()

    blur = filters.Blur_Gaussian("std")
    blur.set_params({"std": 5})
    blur.apply(dat1, rs, named_dims={})
    for j in range(3):
        blur.apply(dat2[:, :, j], rs, named_dims={})
    assert np.array_equal(dat1, dat2)


def test_resolution():
    scale = 3
    height = 50
    width = 50
    rs = np.random.RandomState(seed=42)
    dat1 = rs.randint(low=0, high=255, size=(height, width, 3))
    dat2 = dat1.copy()

    for y in range(height):
        ry = (y // scale) * scale
        for x in range(width):
            rx = (x // scale) * scale
            dat1[y, x, :] = dat1[ry, rx, :]

    res = filters.ResolutionVectorized("scale")
    res.set_params({"scale": scale})
    res.apply(dat2, rs, named_dims={})

    assert np.array_equal(dat1, dat2)


def test_jpeg_compression():
    rs = np.random.RandomState(seed=42)
    dat = rs.randint(low=0, high=255, size=(50, 50))
    orig_dat = np.uint8(dat)

    comp = JPEG_Compression("quality")
    params = {"quality": 50}
    comp.set_params(params)
    comp.apply(dat, rs, named_dims={})
    dat = np.uint8(dat)

    assert not (abs(dat - orig_dat) < 5).all()


def test_sensor_drift():
    drift = SensorDrift("magnitude")
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

    weird = StrangeBehaviour("strange")
    params = {"strange": strange}
    weird.set_params(params)
    y = np.arange(0, 30)
    weird.apply(y, np.random.RandomState(), named_dims={})

    for i in range(15, 21):
        assert y[i] == -300


def test_one_gap():
    gap = Gap("prob_break", "prob_recover", "missing")
    y = np.arange(10000.0)
    params = {"prob_break": 0.0, "prob_recover": 1, "missing": np.nan}
    gap.set_params(params)
    gap.apply(y, np.random.RandomState(), named_dims={})

    for _, val in enumerate(y):
        assert not np.isnan(val)


def test_two_gap():
    gap = Gap("prob_break", "prob_recover", "missing")
    params = {"prob_break": 1.0, "prob_recover": 0.0, "missing": np.nan}
    y = np.arange(10000.0)
    gap.set_params(params)
    gap.apply(y, np.random.RandomState(), named_dims={})

    for _, val in enumerate(y):
        assert np.isnan(val)


def test_apply_with_probability():
    data = np.array([["a"], ["a"], ["a"], ["a"], ["a"], ["a"], ["a"], ["a"], ["a"], ["a"]])

    ocr = OCRError("ps", "p")
    x_node = Array()
    x_node.addfilter(ApplyWithProbability("ocr_node", "ocr_prob"))
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
    x_node.addfilter(Constant('c'))
    out = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.array_equal(out, np.full((5, 5), 5))


def test_identity():
    a = np.arange(25).reshape((5, 5))
    x_node = Array()
    x_node.addfilter(Identity())
    out = x_node.generate_error(a, {}, np.random.RandomState(seed=42))
    assert np.array_equal(out, a)


def test_addition():
    a = np.full((5, 5), 5)
    params = {}
    params['const'] = Constant('c')
    params['c'] = 2
    params['identity'] = Identity()
    x_node = Array()
    x_node.addfilter(Addition('const', 'identity'))
    out = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.array_equal(out, np.full((5, 5), 7))


def test_subtraction():
    a = np.full((5, 5), 5)
    params = {}
    params['const'] = Constant('c')
    params['c'] = 2
    params['identity'] = Identity()
    x_node = Array()
    x_node.addfilter(Subtraction('const', 'identity'))
    out = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.array_equal(out, np.full((5, 5), -3))


def test_multiplication():
    a = np.full((5, 5), 5)
    params = {}
    params['const'] = Constant('c')
    params['c'] = 2
    params['identity'] = Identity()
    x_node = Array()
    x_node.addfilter(Multiplication('const', 'identity'))
    out = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.array_equal(out, np.full((5, 5), 10))


def test_division():
    a = np.full((5, 5), 5.0)
    params = {}
    params['const'] = Constant('c')
    params['c'] = 2
    params['identity'] = Identity()
    x_node = Array()
    x_node.addfilter(Division('const', 'identity'))
    out = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.allclose(out, np.full((5, 5), .4))


def test_integer_division():
    a = np.full((5, 5), 5)
    params = {}
    params['const'] = Constant('c')
    params['c'] = 2
    params['identity'] = Identity()
    x_node = Array()
    x_node.addfilter(IntegerDivision('identity', 'const'))
    out = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.array_equal(out, np.full((5, 5), 2))


def test_modulo():
    a = np.full((5, 5), 5)
    params = {}
    params['const'] = Constant('c')
    params['c'] = 2
    params['identity'] = Identity()
    x_node = Array()
    x_node.addfilter(Modulo('identity', 'const'))
    out = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.array_equal(out, np.full((5, 5), 1))


def test_and():
    a = np.full((5, 5), 5)
    params = {}
    params['const'] = Constant('c')
    params['c'] = 2
    params['identity'] = Identity()
    x_node = Array()
    x_node.addfilter(And('identity', 'const'))
    out = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.array_equal(out, np.full((5, 5), 0))


def test_or():
    a = np.full((5, 5), 5)
    params = {}
    params['const'] = Constant('c')
    params['c'] = 2
    params['identity'] = Identity()
    x_node = Array()
    x_node.addfilter(Or('identity', 'const'))
    out = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.array_equal(out, np.full((5, 5), 7))


def test_xor():
    a = np.full((5, 5), 5)
    params = {}
    params['const'] = Constant('c')
    params['c'] = 3
    params['identity'] = Identity()
    x_node = Array()
    x_node.addfilter(Xor('identity', 'const'))
    out = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.array_equal(out, np.full((5, 5), 6))


def test_difference():
    a = np.full((5, 5), 5)
    params = {}
    params['const'] = Constant('c')
    params['c'] = 2
    params['identity'] = Identity()
    params['addition'] = Addition('identity', 'const')
    x_node = Array()
    x_node.addfilter(Difference("addition"))
    out = x_node.generate_error(a, params, np.random.RandomState(seed=42))

    assert np.array_equal(out, np.full((5, 5), 2))


def test_min():
    a = np.full((5, 5), 5)
    params = {}
    params['const'] = Constant('c')
    params['c'] = 2
    params['identity'] = Identity()
    x_node = Array()
    x_node.addfilter(Min('identity', 'const'))
    out = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.array_equal(out, np.full((5, 5), 2))


def test_max():
    a = np.full((5, 5), 5)
    params = {}
    params['const'] = Constant('c')
    params['c'] = 2
    params['identity'] = Identity()
    x_node = Array()
    x_node.addfilter(Max('identity', 'const'))
    out = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.array_equal(out, np.full((5, 5), 5))


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
    params['filter'] = Identity()
    x_node = Array()
    x_node.addfilter(ModifyAsDataType('dtype', 'filter'))
    out = x_node.generate_error(a, params)
    assert np.array_equal(out, np.array([42]))

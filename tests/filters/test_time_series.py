import numpy as np
from dpemu.nodes import Array
from dpemu.filters.time_series import Gap, SensorDrift


def test_seed_determines_result_for_gap_filter():
    a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    x_node = Array()
    x_node.addfilter(Gap("prob_break", "prob_recover", "missing_value"))
    params = {"prob_break": 0.1, "prob_recover": 0.1, "missing_value": 1337}
    out1 = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    out2 = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.array_equal(out1, out2)


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

import json
import tempfile

import numpy as np

from pytest import approx

from dpemu import pg_utils


def test_load_ocr_error_params():
    temp = tempfile.NamedTemporaryFile(suffix=".json")
    d = {"a": [["a", "o"], [20, 5]],
         "b": [["b", "o"], [8, 2]]}
    temp.write(bytes(json.dumps(d), encoding='UTF-8'))
    temp.seek(0)
    d2 = pg_utils.load_ocr_error_params(temp.name)
    for key, values in d2.items():
        assert key in d
        for i in range(len(values[0])):
            assert values[0][i] == d[key][0][i]
            assert values[1][i] == d[key][1][i]


def test_normalize_ocr_error_params():
    params = {"a": [["a", "o"], [20, 5]],
              "b": [["o", "b"], [2, 8]]}

    normalized_params = pg_utils.normalize_ocr_error_params(params)

    assert normalized_params["a"][0][0] == "a"
    assert normalized_params["a"][0][1] == "o"
    assert normalized_params["a"][1][0] == approx(0.8)
    assert normalized_params["a"][1][1] == approx(0.2)
    assert normalized_params["b"][0][0] == "o"
    assert normalized_params["b"][0][1] == "b"
    assert normalized_params["b"][1][0] == approx(0.2)
    assert normalized_params["b"][1][1] == approx(0.8)


def test_normalize_probs():
    s = sum(pg_utils.normalize_probs([10, 15, 67, 87, 90]))
    assert s == approx(1)


def test_to_time_series_x_y():
    x, y = pg_utils.to_time_series_x_y(np.array([0, 1, 2, 3, 4, 5]), 3)
    assert np.array_equal(x, np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]])) and np.array_equal(y, np.array([3, 4, 5]))

import json
import tempfile

from pytest import approx

import src.problemgenerator.utils as utils


def test_load_ocr_error_params():
    temp = tempfile.NamedTemporaryFile(suffix=".json")
    d = {"a": [["a", "o"], [20, 5]],
         "b": [["b", "o"], [8, 2]]}
    temp.write(bytes(json.dumps(d), encoding='UTF-8'))
    temp.seek(0)
    d2 = utils.load_ocr_error_params(temp.name)
    for key, values in d2.items():
        assert key in d
        for i in range(len(values[0])):
            assert values[0][i] == d[key][0][i]
            assert values[1][i] == d[key][1][i]


def test_normalize_ocr_error_params():
    params = {"a": [["a", "o"], [20, 5]],
              "b": [["b", "o"], [8, 2]]}

    normalized_params = utils.normalize_ocr_error_params(params, 0.5)

    assert normalized_params["a"][0][0] == "a"
    assert normalized_params["a"][0][1] == "o"
    assert normalized_params["a"][1][0] == approx(0.9)
    assert normalized_params["a"][1][1] == approx(0.1)
    assert normalized_params["b"][0][0] == "b"
    assert normalized_params["b"][0][1] == "o"
    assert normalized_params["b"][1][0] == approx(0.9)
    assert normalized_params["b"][1][1] == approx(0.1)


def test_normalize_probs():
    s = sum(utils.normalize_probs([10, 15, 67, 87, 90], 1, 0.5))
    assert s == approx(1)

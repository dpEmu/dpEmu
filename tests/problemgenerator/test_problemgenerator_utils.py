import tempfile
import json
from pytest import approx

import src.problemgenerator.utils as utils


def test_load_ocr_frequencies():
    temp = tempfile.NamedTemporaryFile(suffix=".json")
    d = {"a": [["a", "o"], [20, 5]],
         "b": [["b", "o"], [8, 2]]}
    temp.write(bytes(json.dumps(d), encoding='UTF-8'))
    temp.seek(0)
    d2 = utils.load_ocr_error_frequencies(temp.name)
    for key, values in d2.items():
        assert key in d
        for i in range(len(values[0])):
            assert values[0][i] == d[key][0][i]
            assert values[1][i] == d[key][1][i]


def test_create_normalized_probs():
    params = {"a": [["a", "o"], [20, 5]],
              "b": [["b", "o"], [8, 2]]}

    replacements = utils.create_normalized_probs(params, 0.5)

    assert replacements["a"][0][0] == "o"
    assert replacements["a"][0][1] == "a"
    assert replacements["a"][1][0] == approx(0.1)
    assert replacements["a"][1][1] == approx(0.9)
    assert replacements["b"][0][0] == "o"
    assert replacements["b"][0][1] == "b"
    assert replacements["b"][1][0] == approx(0.1)
    assert replacements["b"][1][1] == approx(0.9)


def test_normalize():
    s = sum(utils.normalize([10, 15, 67, 87, 90], 0.5))
    assert s == approx(1)

import tempfile
import json
import numpy as np
from PIL import Image
import run


def test_do_replacements():
    command = "Hello World Hello"
    replacements = {"Hello": "Terve", "World": "Maailma"}

    assert run.do_replacements(command, replacements) == "Terve Maailma Terve"


def test_read_commands():
    temp = tempfile.NamedTemporaryFile()
    temp.write(b"Hello\nWorld")
    temp.seek(0)
    assert run.read_commands_file(temp.name) == ("Hello", "World")


def test_read_analyzer_files():
    temp1 = tempfile.NamedTemporaryFile(suffix=".json")
    temp1.write(bytes(json.dumps({"A": 1, "B": 2}), encoding='UTF-8'))
    temp1.seek(0)
    temp2 = tempfile.NamedTemporaryFile(suffix=".txt")
    temp3 = tempfile.NamedTemporaryFile(suffix=".png")
    array = np.zeros([4, 5, 3], dtype=np.uint8)
    array[:, :] = [255, 128, 0]
    Image.fromarray(array).save(temp3.name)
    temp3.seek(0)
    temp4 = tempfile.NamedTemporaryFile(suffix=".pdf")
    li = run.read_analyzer_files([temp1.name, temp2.name, temp3.name, temp4.name])

    assert len(li) == 2

import tempfile
from mock import patch
import run


def test_do_replacements():
    command = "Hello World Hello"
    replacements = {"Hello" : "Terve", "World" : "Maailma"}

    assert run.do_replacements(command, replacements) == "Terve Maailma Terve"

def test_read_commands():
    temp = tempfile.NamedTemporaryFile()
    temp.write(b"Hello\nWorld")
    temp.seek(0)
    assert run.read_commands_file(temp.name) == ("Hello", "World")

def test_create_replacements():
    assert True

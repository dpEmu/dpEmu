from mock import patch
import numpy as np
import src.utils as utils

'''
@patch("datetime.datetime.now().strftime", return_value="1")
@patch("src.utils.get_project_root", return_value="root")
def test_generate_unique_path(choices_mock):
    assert utils.generate_unique_path("tmp", "txt") == "root/tmp/1.txt"
'''

@patch("src.utils.get_project_root", return_value="root")
def test_get_project_root(root_mock):
    assert utils.get_project_root() == "root"

def test_expand_parameter_to_linspace():
    param1 = [2.0]
    param2 = [1.0, 50.0]
    assert np.array_equal(utils.expand_parameter_to_linspace(param1), np.array([2.0]))
    assert np.array_equal(utils.expand_parameter_to_linspace(param2), np.array([i+1 for i in range(50)]))

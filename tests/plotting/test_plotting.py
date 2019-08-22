import re
import numpy as np
from dpemu.nodes import Array
from dpemu.nodes import Series, TupleSeries
from dpemu.filters import Addition, Constant
from dpemu.filters.common import Missing
from dpemu import plotting_utils


def test_visualizing_array_node():
    x_node = Array()
    path = plotting_utils.visualize_error_generator(x_node, False)
    file = open(path, 'r')
    data = file.read()
    assert re.compile(r'1.*Array').search(data)


def test_visualizing_series_and_array_nodes():
    x_node = Array()
    series_node = Series(x_node)
    path = plotting_utils.visualize_error_generator(series_node, False)
    file = open(path, 'r')
    data = file.read()
    assert re.compile(r'1.*Series').search(data)
    assert re.compile(r'1 -> 2').search(data)


def test_visualizing_tuple_series_and_two_array_nodes():
    x_node = Array()
    y_node = Array()
    series_node = TupleSeries([x_node, y_node])
    path = plotting_utils.visualize_error_generator(series_node, False)
    file = open(path, 'r')
    data = file.read()
    assert re.compile(r'1.*TupleSeries').search(data)
    assert re.compile(r'1 -> 2').search(data)
    assert re.compile(r'1 -> 3').search(data)


def test_visualizing_array_node_with_filter():
    x_node = Array()
    x_node.addfilter(Missing("p", "missing_value"))
    tree = x_node.get_parametrized_tree({'p': 0.5, 'missing_value': np.nan})
    path = plotting_utils.visualize_error_generator(tree, False)
    file = open(path, 'r')
    data = file.read()
    assert re.compile(r'2.*Missing.*probability: 0').search(data)
    assert re.compile(r'1 -> 2').search(data)


def test_visualizing_array_node_with_complex_filter():
    x_node = Array()
    const = Constant("c")
    addition = Addition(const, const)
    x_node.addfilter(addition)
    path = plotting_utils.visualize_error_generator(
        x_node.get_parametrized_tree({'c': 5}), False)
    file = open(path, 'r')
    data = file.read()
    assert re.compile(r'2.*Addition').search(data)
    assert re.compile(r'3.*Constant.*value: 5').search(data)
    assert re.compile(r'4.*Constant.*value: 5').search(data)
    assert re.compile(r'2 -> 3.*filter_a').search(data)
    assert re.compile(r'2 -> 4.*filter_b').search(data)

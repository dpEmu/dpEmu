import re
from dpemu import array
from dpemu import filters
from dpemu import series
from dpemu import plotting_utils


def test_visualizing_array_node():
    x_node = array.Array()
    path = plotting_utils.visualize_error_generator(x_node, False)
    file = open(path, 'r')
    data = file.read()
    assert re.compile(r'1.*Array').search(data)


def test_visualizing_series_and_array_nodes():
    x_node = array.Array()
    series_node = series.Series(x_node)
    path = plotting_utils.visualize_error_generator(series_node, False)
    file = open(path, 'r')
    data = file.read()
    assert re.compile(r'1.*Series').search(data)
    assert re.compile(r'1 -> 2').search(data)


def test_visualizing_tuple_series_and_two_array_nodes():
    x_node = array.Array()
    y_node = array.Array()
    series_node = series.TupleSeries([x_node, y_node])
    path = plotting_utils.visualize_error_generator(series_node, False)
    file = open(path, 'r')
    data = file.read()
    assert re.compile(r'1.*TupleSeries').search(data)
    assert re.compile(r'1 -> 2').search(data)
    assert re.compile(r'1 -> 3').search(data)


def test_visualizing_array_node_with_filter():
    x_node = array.Array()
    x_node.addfilter(filters.Missing("p"))
    path = plotting_utils.visualize_error_generator(x_node.get_parametrized_tree({'p': 0.5}), False)
    file = open(path, 'r')
    data = file.read()
    assert re.compile(r'2.*Missing.*probability: 0').search(data)
    assert re.compile(r'1 -> 2').search(data)


def test_visualizing_array_node_with_complex_filter():
    x_node = array.Array()
    addition = filters.Addition("f1", "f2")
    const = filters.Constant("c")
    x_node.addfilter(addition)
    path = plotting_utils.visualize_error_generator(
        x_node.get_parametrized_tree({'f1': const, 'f2': const, 'c': 5}), False)
    file = open(path, 'r')
    data = file.read()
    assert re.compile(r'2.*Addition').search(data)
    assert re.compile(r'3.*Constant.*value: 5').search(data)
    assert re.compile(r'4.*Constant.*value: 5').search(data)
    assert re.compile(r'2 -> 3.*filter_a').search(data)
    assert re.compile(r'2 -> 4.*filter_b').search(data)

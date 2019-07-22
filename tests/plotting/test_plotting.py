import re
import src.problemgenerator.array as array
import src.problemgenerator.filters as filters
import src.problemgenerator.series as series
from src.plotting.utils import visualize_error_generator


def test_visualizing_array_node():
    x_node = array.Array((5))
    path = visualize_error_generator(x_node, False)
    file = open(path, 'r')
    data = file.read()
    assert re.compile(r'1.*Array').search(data)


def test_visualizing_series_and_array_nodes():
    x_node = array.Array((5))
    series_node = series.Series(x_node)
    path = visualize_error_generator(series_node, False)
    file = open(path, 'r')
    data = file.read()
    assert re.compile(r'1.*Series').search(data)
    assert re.compile(r'1 -> 2').search(data)


def test_visualizing_tuple_series_and_two_array_nodes():
    x_node = array.Array((5))
    y_node = array.Array((5))
    series_node = series.TupleSeries([x_node, y_node])
    path = visualize_error_generator(series_node, False)
    file = open(path, 'r')
    data = file.read()
    assert re.compile(r'1.*TupleSeries').search(data)
    assert re.compile(r'1 -> 2').search(data)
    assert re.compile(r'1 -> 3').search(data)


def test_visualizing_array_node_with_filter():
    x_node = array.Array((5))
    x_node.addfilter(filters.Missing("p"))
    path = visualize_error_generator(x_node.get_parametrized_tree({'p': 0.5}), False)
    file = open(path, 'r')
    data = file.read()
    assert re.compile(r'2.*Missing.*probability: 0').search(data)
    assert re.compile(r'1 -> 2').search(data)


def test_visualizing_array_node_with_complex_filter():
    x_node = array.Array((5))
    addition = filters.Addition("f1", "f2")
    const = filters.Constant("c")
    x_node.addfilter(addition)
    path = visualize_error_generator(x_node.get_parametrized_tree({'f1': const, 'f2': const, 'c': 5}), False)
    file = open(path, 'r')
    data = file.read()
    assert re.compile(r'2.*Addition').search(data)
    assert re.compile(r'3.*Constant.*value: 5').search(data)
    assert re.compile(r'4.*Constant.*value: 5').search(data)
    assert re.compile(r'2 -> 3.*filter_a').search(data)
    assert re.compile(r'2 -> 4.*filter_b').search(data)

# MIT License
#
# Copyright (c) 2019 Tuomas Halvari, Juha Harviainen, Juha Mylläri, Antti Röyskö, Juuso Silvennoinen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
    addition = Addition("f1", "f2")
    const = Constant("c")
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

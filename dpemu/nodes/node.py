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

import numpy as np
import copy
from abc import ABC, abstractmethod


class Node(ABC):
    """Node is the superclass for all node classes of the error generation tree.
    """

    def __init__(self, children):
        """
        Args:
            children (list): A list of all child nodes of the node.
        """
        self.filters = []
        self.children = children

    def addfilter(self, error_filter):
        """Attach a filter (error source) to the node.

        Args:
            error_filter (object): A pre-existing or user-specified filter
        """
        self.filters.append(error_filter)

    def set_error_params(self, params_dict):
        """Set error parameters for the filter.

        Args:
            params_dict (dict): A Python dictionary.
        """
        for filter_ in self.filters:
            filter_.set_params(params_dict)
        for child in self.children:
            child.set_error_params(params_dict)

    @abstractmethod
    def process(self, data, random_state, index_tuple=(), named_dims={}):
        """Processes the given data by passing it recursively in the error generation tree and applying filters to it.

        Args:
            data (numpy.ndarray): Data to be modified as a Numpy array.
            random_state (mtrand.RandomState): An instance of mtrand.RandomState to ensure repeatability.
            index_tuple (tuple, optional): The index of the node. Defaults to ().
            named_dims (dict, optional): Named dimensions. Defaults to {}.
        """
        pass

    def generate_error(self, data, error_params, random_state=np.random.RandomState(42)):
        """Returns the data with the desired errors introduced.

        The original data object is not modified. The error parameters must be provided as
        a dictionary whose keys are the parameter identifiers (given as parameters to the
        filters) and whose values are the desired parameter values.

        Args:
            data (numpy.ndarray): Data to be modified as a Numpy array.
            error_params (dict): A dictionary containing the parameters for error generation.
            random_state (mtrand.RandomState, optional): An instance of numpy.random.RandomState.
                Defaults to np.random.RandomState(42).

        Returns:
            numpy.ndarray: Errorified data.
        """
        copy_data = copy.deepcopy(data)
        copy_tree = copy.deepcopy(self)
        copy_tree.set_error_params(error_params)
        copy_tree.process(copy_data, random_state)
        return copy_data

    def get_parametrized_tree(self, error_params):
        """Returns an error generation tree with desired parameter values of the filters.

        Args:
            error_params (dict): A dictionary containing the parameters for error generation.

        Returns:
            Node: A root node of the error generation tree.
        """
        copy_tree = copy.deepcopy(self)
        copy_tree.set_error_params(error_params)
        return copy_tree


class LeafNode(Node):
    """LeafNode is the superclass for all leaf node classes of the error generation tree.
    """

    def __init__(self):
        super().__init__([])

    def apply_filters(self, node_data, random_state, named_dims):
        for f in self.filters:
            f.apply(node_data, random_state, named_dims)


def get_node_data(data, index_tuple, make_array=True):
    """Returns some desired subset of the data to the node as well as additional information about its structure.

    Args:
        data (obj): The original data the node received.
        index_tuple (tuple, optional): The index of the node. Defaults to ().
        make_array (bool, optional): If True, the data array is typecasted to numpy.ndarray. Defaults to True.

    Returns:
        numpy.ndarray, bool, bool, bool: Data as a numpy array and bools telling if the data is
            a list, a scalar or a tuple.
    """
    index_list = list(index_tuple)
    while index_list:
        data = data[index_list.pop(0)]
    node_data_is_list = type(data) is list
    node_data_is_scalar = np.isscalar(data)
    node_data_is_tuple = type(data) is tuple
    if make_array and type(data) is not np.ndarray:
        data = np.array(data)
    return data, node_data_is_list, node_data_is_scalar, node_data_is_tuple


def assign(data_root, index_tuple, value):
    """Makes the assignment data_root[index_tuple] = value.
    """
    if not index_tuple:
        if len(value) == len(data_root):
            data_root[:] = value[:]
            return
        else:
            raise Exception(f"""Cannot assign value
                                to data root
                                """)
    location = data_root
    while len(index_tuple) > 1 and type(location) is not np.ndarray:
        location = location[index_tuple[0]]
        index_tuple = index_tuple[1:]
    if len(index_tuple) == 1:
        location[index_tuple[0]] = value
    else:
        location[index_tuple] = value

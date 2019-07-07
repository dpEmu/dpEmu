import numpy as np


class Node:

    def __init__(self, children):
        self.filters = []
        self.children = children
        self.shape = ()

    def addfilter(self, error_filter):
        """Attach a filter (error source) to the node."""
        self.filters.append(error_filter)
        error_filter.shape = self.shape

    def set_error_params(self, params_dict):
        for filter_ in self.filters:
            filter_.set_params(params_dict)
        for child in self.children:
            child.set_error_params(params_dict)


def get_node_data(data, index_tuple, make_array=True):
    index_list = list(index_tuple)
    while index_list:
        data = data[index_list.pop(0)]
    node_data_is_list = type(data) is list
    node_data_is_scalar = np.isscalar(data)
    if make_array and type(data) is not np.ndarray:
        data = np.array(data)
    return data, node_data_is_list, node_data_is_scalar
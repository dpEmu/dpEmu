import numpy as np
import copy


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

    def process(self, data, random_state):
        pass

    def generate_error(self, data, error_params, random_state=np.random.RandomState(42)):
        """Returns the data with the desired errors introduced. The original
        data object is not modified. The error parameters must be provided as
        a dictionary whose keys are the parameter identifiers (given as
        parameters to the filters) and whose values are the desired parameter
        values.
        """
        self.set_error_params(error_params)
        copy_data = copy.deepcopy(data)
        self.process(copy_data, random_state)
        return copy_data


class LeafNode(Node):
    """LeafNode is the superclass for all leaf node classes of the error generation tree.
    """

    def __init__(self):
        super().__init__([])

    def apply_filters(self, node_data, random_state, named_dims):
        for f in self.filters:
            f.apply(node_data, random_state, named_dims)


def get_node_data(data, index_tuple, make_array=True):
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

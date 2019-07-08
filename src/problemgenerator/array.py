import numpy as np
from src.problemgenerator.node import Node, get_node_data


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


class Array(Node):
    """An Array node represents a data array of any dimension (>= 0).
    One or more filters (error sources) can be added to the node.
    The filters are applied in the order in which they are added.
    """

    def __init__(self, shape=()):
        self.shape = shape
        super().__init__([])

    def apply_filters(self, node_data, random_state, named_dims):
        for f in self.filters:
            f.apply(node_data, random_state, named_dims)

    def process(self, data, random_state, index_tuple=(), named_dims={}):
        """Apply all filters in this node."""
        node_data, is_list, is_scalar = get_node_data(data, index_tuple)
        if is_list:
            self.apply_filters(node_data, random_state, named_dims)
            assign(data, index_tuple, list(node_data))
        elif is_scalar:
            self.apply_filters(node_data, random_state, named_dims)
            assign(data, index_tuple, node_data[()])
        else:
            self.apply_filters(node_data, random_state, named_dims)

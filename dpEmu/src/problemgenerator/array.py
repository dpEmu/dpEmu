from src.problemgenerator.node import LeafNode, get_node_data, assign


class Array(LeafNode):

    """An Array node represents a data array of any dimension (>= 0).

    One or more filters (error sources) can be added to the node.
    The filters are applied in the order in which they are added.

    Args:
        Node (object):
    """

    def __init__(self):
        super().__init__()

    def apply_filters(self, node_data, random_state, named_dims):
        """Apply filters to data contained in this array.

        Args:
            node_data (numpy.ndarray): Data to be modified as a Numpy array.
            random_state (mtrand.RandomState): An instance of numpy.random.RandomState.
            named_dims (dict): Named dimensions.
        """
        for f in self.filters:
            f.apply(node_data, random_state, named_dims)

    def process(self, data, random_state, index_tuple=(), named_dims={}):
        """Apply all filters in this node.

        Args:
            data ([type]): [description]
            random_state (mtrand.RandomState): An instance of numpy.random.RandomState
            index_tuple (tuple, optional): The index of the node. Defaults to ().
            named_dims (dict, optional): Named dimensions. Defaults to {}.
        """
        node_data, is_list, is_scalar, is_tuple = get_node_data(data, index_tuple)
        if is_list:
            self.apply_filters(node_data, random_state, named_dims)
            assign(data, index_tuple, list(node_data))
        elif is_scalar:
            self.apply_filters(node_data, random_state, named_dims)
            assign(data, index_tuple, node_data[()])
        elif is_tuple:
            self.apply_filters(node_data, random_state, named_dims)
            assign(data, index_tuple, tuple(node_data))
        else:
            self.apply_filters(node_data, random_state, named_dims)


class TupleArray(Array):
    """A tuple consisting of Arrays.

    Args:
        Array (object): An Array node represents a data array of any dimension (>= 0).
    """

    def __init__(self):
        super().__init__([])

    def process(self, data, random_state, index_tuple=(), named_dims={}):
        pass

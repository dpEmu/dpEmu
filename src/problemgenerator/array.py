from src.problemgenerator.node import LeafNode, get_node_data, assign


class Array(LeafNode):
    """An Array node represents a data array of any dimension (>= 0).
    One or more filters (error sources) can be added to the node.
    The filters are applied in the order in which they are added.
    """

    def __init__(self, shape=()):
        self.shape = shape
        super().__init__()

    def apply_filters(self, node_data, random_state, named_dims):
        for f in self.filters:
            f.apply(node_data, random_state, named_dims)

    def process(self, data, random_state, index_tuple=(), named_dims={}):
        """Apply all filters in this node."""
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

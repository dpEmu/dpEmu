from src.problemgenerator.node import LeafNode, get_node_data, assign


class Tuple(LeafNode):

    def __init__(self):
        super().__init__()

    def process(self, data, random_state, index_tuple=(), named_dims={}):
        """Apply all filters in this node."""
        node_data, _, _, _ = get_node_data(data, index_tuple)
        self.apply_filters(node_data, random_state, named_dims)
        assign(data, index_tuple, tuple(node_data))

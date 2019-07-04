from src.problemgenerator.node import Node

class Array(Node):
    """An Array node represents a data array of any dimension (>= 0).
    One or more filters (error sources) can be added to the node.
    The filters are applied in the order in which they are added.
    """

    def __init__(self, shape=()):
        self.shape = shape
        super().__init__([])

    def process(self, data, random_state, index_tuple=(), named_dims={}):
        """Apply all filters in this node and its descendent nodes."""
        for f in self.filters:
            print(f"calling filter with random state {random_state}, ind tuple {index_tuple}")
            f.apply(data, random_state, index_tuple, named_dims)

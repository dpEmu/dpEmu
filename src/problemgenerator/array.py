class Array:
    """An Array node represents a data array of any dimension (>= 0).
    One or more filters (error sources) can be added to the node.
    The filters are applied in the order in which they are added.
    """
    
    def __init__(self, shape):
        self.shape = shape
        self.filters = []

    def addfilter(self, error_filter):
        """Attach a filter (error source) to the node."""
        self.filters.append(error_filter)
        error_filter.shape = self.shape

    def process(self, data, random_state, index_tuple=(), named_dims={}):
        """Apply all filters in this node and its descendent nodes."""
        for f in self.filters:
            f.apply(data, random_state, index_tuple, named_dims)

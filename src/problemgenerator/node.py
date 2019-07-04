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

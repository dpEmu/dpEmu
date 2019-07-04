from src.problemgenerator.node import Node
class Series(Node):

    def __init__(self, child, dim_name=None):
        super.__init__([child])
        self.dim_name = dim_name
        self.filters = []

    def addfilter(self, custom_filter):
        self.filters.append(custom_filter)

    def process(self, data, random_state, index_tuple=(), named_dims={}):
        data_length = data[index_tuple].shape[0]
        for i in range(data_length):
            if self.dim_name:
                named_dims[self.dim_name] = i
            self.children[0].process(data, random_state, (i, *index_tuple), named_dims)


class TupleSeries:

    def __init__(self, children, dim_name=None):
        self.children = children
        self.dim_name = dim_name

    def process(self, data, random_state, named_dims={}):
        data_length = data[0].shape[0]
        for i, child in enumerate(self.children):
            for j in range(data_length):
                if self.dim_name:
                    named_dims[self.dim_name] = j
                child.process(data[i], random_state, (j,), named_dims)

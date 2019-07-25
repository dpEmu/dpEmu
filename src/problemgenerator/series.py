from src.problemgenerator.node import Node, get_node_data
from src.problemgenerator.utils import first_dimension_length


class Series(Node):

    def __init__(self, child, dim_name=None):
        super().__init__([child])
        self.dim_name = dim_name
        self.filters = []

    def addfilter(self, custom_filter):
        self.filters.append(custom_filter)

    def process(self, data, random_state, index_tuple=(), named_dims={}):
        node_data, _, _, _ = get_node_data(data, index_tuple, make_array=False)
        data_length = first_dimension_length(node_data)
        for i in range(data_length):
            if self.dim_name:
                named_dims[self.dim_name] = i
            self.children[0].process(data, random_state, (i, *index_tuple), named_dims)


class TupleSeries(Node):

    def __init__(self, children, dim_name=None):
        super().__init__(children)
        self.dim_name = dim_name

    def process(self, data, random_state, index_tuple=(), named_dims={}):
        node_data = get_node_data(data, index_tuple, make_array=False)[0]
        data_length = first_dimension_length(node_data[0])
        for i, child in enumerate(self.children):
            for j in range(data_length):
                if self.dim_name:
                    named_dims[self.dim_name] = j
                child.process(data[i], random_state, (j,), named_dims)

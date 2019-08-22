# MIT License
#
# Copyright (c) 2019 Tuomas Halvari, Juha Harviainen, Juha Mylläri, Antti Röyskö, Juuso Silvennoinen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from .node import Node, get_node_data
from ..pg_utils import first_dimension_length


class Series(Node):
    """[summary]

    [extended_summary]

    Args:
        Node ([type]): [description]
    """

    def __init__(self, child, dim_name=None):
        """
        Args:
            child ([type]): [description]
            dim_name ([type], optional): [description]. Defaults to None.
        """
        super().__init__([child])
        self.dim_name = dim_name

    def process(self, data, random_state, index_tuple=(), named_dims={}):
        """[summary]

        [extended_summary]

        Args:
            data ([type]): [description]
            random_state ([type]): [description]
            index_tuple (tuple, optional): [description]. Defaults to ().
            named_dims (dict, optional): [description]. Defaults to {}.
        """
        node_data, _, _, _ = get_node_data(data, index_tuple, make_array=False)
        data_length = first_dimension_length(node_data)
        for i in range(data_length):
            if self.dim_name:
                named_dims[self.dim_name] = i
            self.children[0].process(data, random_state, (i, *index_tuple), named_dims)


class TupleSeries(Node):
    """[summary]

    [extended_summary]

    Args:
        Node ([type]): [description]
    """

    def __init__(self, children, dim_name=None):
        """
        Args:
            children ([type]): [description]
            dim_name ([type], optional): [description]. Defaults to None.
        """
        super().__init__(children)
        self.dim_name = dim_name

    def process(self, data, random_state, index_tuple=(), named_dims={}):
        """[summary]

        [extended_summary]

        Args:
            data ([type]): [description]
            random_state ([type]): [description]
            index_tuple (tuple, optional): [description]. Defaults to ().
            named_dims (dict, optional): [description]. Defaults to {}.
        """
        node_data = get_node_data(data, index_tuple, make_array=False)[0]
        data_length = first_dimension_length(node_data[0])
        for i, child in enumerate(self.children):
            for j in range(data_length):
                if self.dim_name:
                    named_dims[self.dim_name] = j
                child.process(data[i], random_state, (j,), named_dims)

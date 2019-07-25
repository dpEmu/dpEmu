import numpy as np
import copy


class Node:
    """[summary]

    [extended_summary]
    """

    def __init__(self, children):
        """
        Args:
            children ([type]): [description]
        """
        self.filters = []
        self.children = children
        self.shape = ()

    def addfilter(self, error_filter):
        """Attach a filter (error source) to the node.

        Args:
            error_filter (object): A pre-existing or user-specified filter
        """
        self.filters.append(error_filter)
        error_filter.shape = self.shape

    def set_error_params(self, params_dict):
        """Set error parameters for the filter.

        Args:
            params_dict (dict): A Python dictionary.
        """
        for filter_ in self.filters:
            filter_.set_params(params_dict)
        for child in self.children:
            child.set_error_params(params_dict)

    def process(self, data, random_state):
        pass

    def generate_error(self, data, error_params, random_state=np.random.RandomState(42)):
        """Returns the data with the desired errors introduced.

        The original data object is not modified. The error parameters must be provided as
        a dictionary whose keys are the parameter identifiers (given as parameters to the
        filters) and whose values are the desired parameter values.

        Args:
            data ([type]): [description]
            error_params ([type]): [description]
            random_state (mtrand.RandomState, optional): An instance of numpy.random.RandomState.
                Defaults to np.random.RandomState(42).

        Returns:
            [type]: [description]
        """
        copy_data = copy.deepcopy(data)
        copy_tree = copy.deepcopy(self)
        copy_tree.set_error_params(error_params)
        copy_tree.process(copy_data, random_state)
        return copy_data

    def get_parametrized_tree(self, error_params):
        """[summary]

        [extended_summary]

        Args:
            error_params ([type]): [description]

        Returns:
            [type]: [description]
        """
        copy_tree = copy.deepcopy(self)
        copy_tree.set_error_params(error_params)
        return copy_tree


def get_node_data(data, index_tuple, make_array=True):
    """[summary]

    [extended_summary]

    Args:
        data ([type]): [description]
        index_tuple ([type]): [description]
        make_array (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    index_list = list(index_tuple)
    while index_list:
        data = data[index_list.pop(0)]
    node_data_is_list = type(data) is list
    node_data_is_scalar = np.isscalar(data)
    if make_array and type(data) is not np.ndarray:
        data = np.array(data)
    return data, node_data_is_list, node_data_is_scalar

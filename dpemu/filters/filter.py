import random
import numpy as np
from abc import ABC, abstractmethod
from ..pg_utils import generate_random_dict_key


class Filter(ABC):
    """A Filter is an error source which can be attached to an Array node.

    The apply method applies the filter. A filter may always assume that
    it is acting upon a NumPy array. (When the underlying data object is not
    a NumPy array, the required conversions are performed by the Array node
    to which the Filter is attached.)

    Args:
        ABC (object): Helper class that provides a standard way to create an ABC using
    inheritance.
    """

    def __init__(self):
        """Set the seeds for the RNG's of NumPy and Python.
        """
        np.random.seed(42)
        random.seed(42)

    @abstractmethod
    def set_params(self, params_dict):
        """Set parameters for error generation.

        Args:
            params_dict (dict): A dictionary which contains error parameter name and value pairs.
        """
        pass

    @abstractmethod
    def apply(self, node_data, random_state, named_dims):
        """Modifies the data according to the functionality of the filter.

        Args:
            node_data (numpy.ndarray): Data to be modified as a NumPy array.
            random_state (mtrand.RandomState): An instance of numpy.random.RandomState() random number generator.
            named_dims (dict): Named dimensions.
        """
        pass


class Constant(Filter):
    """[summary]

    [extended_summary]

    Inherits Filter class.
    """

    def __init__(self, value_id):
        super().__init__()
        self.value_id = value_id

    def set_params(self, params_dict):
        self.value = params_dict[self.value_id]

    def apply(self, node_data, random_state, named_dims):
        node_data.fill(self.value)


class Identity(Filter):
    """This filter acts as the identity operator and does not modify data.

    Inherits Filter class.
    """

    def __init__(self):
        super().__init__()

    def set_params(self, params_dict):
        pass

    def apply(self, node_data, random_state, named_dims):
        pass


class BinaryFilter(Filter):
    """This abstract filter takes two filters and applies some pairwise binary operation on their results.

    Inherits Filter class.
    """

    def __init__(self, filter_a_id, filter_b_id):
        """
        Args:
            filter_a_id (str): A key which maps to the first filter
            filter_b_id (str): A key which maps to the second filter
        """
        super().__init__()
        self.filter_a_id = filter_a_id
        self.filter_b_id = filter_b_id

    def apply(self, node_data, random_state, named_dims):
        data_a = node_data.copy()
        data_b = node_data.copy()
        self.filter_a.apply(data_a, random_state, named_dims)
        self.filter_b.apply(data_b, random_state, named_dims)
        for index, _ in np.ndenumerate(node_data):
            node_data[index] = self.operation(data_a[index], data_b[index])

    def set_params(self, params_dict):
        self.filter_a = params_dict[self.filter_a_id]
        self.filter_b = params_dict[self.filter_b_id]
        self.filter_a.set_params(params_dict)
        self.filter_b.set_params(params_dict)

    @abstractmethod
    def operation(self, element_a, element_b):
        """The operation which is applied pairwise on the n-dimensional arrays of child filters.

        Args:
            element_a (object): The first element
            element_b (object): The second element
        """
        pass


class Addition(BinaryFilter):
    """This filter does pairwise addition on the multidimensional arrays returned by the child filters.

    Inherits BinaryFilter class.
    """

    def operation(self, element_a, element_b):
        return element_a + element_b


class Subtraction(BinaryFilter):
    """This filter does pairwise subtraction on the multidimensional arrays returned by the child filters.

    Inherits BinaryFilter class.
    """

    def operation(self, element_a, element_b):
        return element_a - element_b


class Multiplication(BinaryFilter):
    """This filter does pairwise multiplication on the multidimensional arrays returned by the child filters.

    Inherits BinaryFilter class.
    """

    def operation(self, element_a, element_b):
        return element_a * element_b


class Division(BinaryFilter):
    """This filter does pairwise division on the multidimensional arrays returned by the child filters.

    Inherits BinaryFilter class.
    """

    def operation(self, element_a, element_b):
        return element_a / element_b


class IntegerDivision(BinaryFilter):
    """This filter does pairwise integer division on the multidimensional arrays returned by the child filters.

    Inherits BinaryFilter class.
    """

    def operation(self, element_a, element_b):
        return element_a // element_b


class Modulo(BinaryFilter):
    """This filter does pairwise modulo operation on the multidimensional arrays returned by the child filters.

    Inherits BinaryFilter class.
    """

    def operation(self, element_a, element_b):
        return element_a % element_b


class And(BinaryFilter):
    """This filter does pairwise bitwise AND on the multidimensional arrays returned by the child filters.

    Inherits BinaryFilter class.
    """

    def operation(self, element_a, element_b):
        return element_a & element_b


class Or(BinaryFilter):
    """"This filter does pairwise bitwise OR on the multidimensional arrays returned by the child filters.

    Inherits BinaryFilter class.
    """

    def operation(self, element_a, element_b):
        return element_a | element_b


class Xor(BinaryFilter):
    """This filter does pairwise bitwise XOR on the multidimensional arrays returned by the child filters.

    Inherits BinaryFilter class.
    """

    def operation(self, element_a, element_b):
        return element_a ^ element_b


class Difference(Filter):
    """This filter returns the difference between the original and the filtered data,
    i.e. it is shorthand for Subtraction(filter, Identity()).

    Inherits BinaryFilter class.
    """

    def __init__(self, ftr_id):
        super().__init__()
        self.ftr_id = ftr_id

    def set_params(self, params_dict):
        identity_key = generate_random_dict_key(params_dict, "identity")
        params_dict[identity_key] = Identity()
        self.ftr = Subtraction(self.ftr_id, identity_key)
        self.ftr.set_params(params_dict)

    def apply(self, node_data, random_state, named_dims):
        self.ftr.apply(node_data, random_state, named_dims)


class Max(BinaryFilter):
    """This filter returns a multidimensional array of pairwise maximums
    of the multidimensional arrays returned by the child filters.

    Inherits BinaryFilter class.
    """

    def operation(self, element_a, element_b):
        return max(element_a, element_b)


class Min(BinaryFilter):
    """This filter returns a multidimensional array of pairwise minimums
    of the multidimensional arrays returned by the child filters.

    Inherits BinaryFilter class.
    """

    def operation(self, element_a, element_b):
        return min(element_a, element_b)

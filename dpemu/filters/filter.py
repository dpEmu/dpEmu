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

import random
import numpy as np
from abc import ABC, abstractmethod


class Filter(ABC):
    """A Filter is an error source which can be attached to an Array node.

    The apply method applies the filter to the data. A filter may always assume that
    it is acting upon a NumPy array. (if the underlying data object is not
    a NumPy array, the required conversions are performed by the Array node
    to which the Filter is attached.)

    Args:
        ABC (object): Helper class that provides a standard way to create
    an abstract class using inheritance.
    """

    # TODO: should this really be done here??
    def __init__(self):
        """Set the seeds for the RNG's of NumPy and Python.
        """
        np.random.seed(42)
        random.seed(42)

    def set_params(self, params_dict):
        """Set parameters for error generation.

        Args:
            params_dict (dict): A dictionary containing key-value pairs of error parameters.
        """
        original = self.__dict__.copy()
        for key in original:
            if key[-3:] == "_id":
                value = self.__dict__[key]
                if value is not None:
                    try:
                        self.__dict__[key[:-3]] = params_dict[value]
                    except KeyError as e:
                        message = "The error parameter dictionary does not contain a parameter "\
                                  f"with the identifier '{value}', which is expected by "\
                                  f"the Filter {self}."
                        raise Exception(message) from e

        for key in self.__dict__:
            value = self.__dict__[key]
            if isinstance(value, Filter):
                value.set_params(params_dict)

    @abstractmethod
    def apply(self, node_data, random_state, named_dims):
        """Applies the filter to the data.

        Args:
            node_data (numpy.ndarray): Data to be modified as a NumPy array.
            random_state (mtrand.RandomState): An instance of numpy.random.RandomState() random number generator.
            named_dims (dict): Named dimensions.
        """
        pass


# TODO: "Inherits Filter class" -> "Inherits the Filter-class" ?
class Constant(Filter):
    """Overwrites all values in the data with the given value.

    Inherits Filter class.
    """

    def __init__(self, value_id):
        """
        Args:
            value_id: The key mapping to the value to overwrite values in the data with.
        """
        super().__init__()
        self.value_id = value_id

    def apply(self, node_data, random_state, named_dims):
        node_data.fill(self.value)


# TODO: Isn't this just the base Filter class?
class Identity(Filter):
    """Acts as the identity operator, thus doesn't modify the data.

    Inherits Filter class.
    """

    def __init__(self):
        super().__init__()

    def apply(self, node_data, random_state, named_dims):
        pass


class BinaryFilter(Filter):
    """Abstract Filter applying two given filters to the data, combining the results with a pairwise binary operation.

    The pairwise binary operation is specified by the inheriting class by overriding the operation-function.

    Inherits Filter class.
    """

    def __init__(self, filter_a, filter_b):
        """
        Args:
            filter_a (str): The first filter.
            filter_b (str): The second filter.
        """
        super().__init__()
        self.filter_a = filter_a
        self.filter_b = filter_b

    def apply(self, node_data, random_state, named_dims):
        data_a = node_data.copy()
        data_b = node_data.copy()
        self.filter_a.apply(data_a, random_state, named_dims)
        self.filter_b.apply(data_b, random_state, named_dims)
        for index, _ in np.ndenumerate(node_data):
            node_data[index] = self.operation(data_a[index], data_b[index])

    @abstractmethod
    def operation(self, element_a, element_b):
        """The pairwise binary operation used to combine results from the two child filters.

        Args:
            element_a (object): The element from the data filter_a operated on.
            element_b (object): The element from the data filter_b operated on.
        """
        pass


class Addition(BinaryFilter):
    """Combines results of the two child filters by adding them together.

    Inherits BinaryFilter class.
    """

    def operation(self, element_a, element_b):
        return element_a + element_b


class Subtraction(BinaryFilter):
    """Combines results of the two child filters by subtracting the results of the second from the firsts.

    Inherits BinaryFilter class.
    """

    def operation(self, element_a, element_b):
        return element_a - element_b


class Multiplication(BinaryFilter):
    """Combines results of the two child filters by multiplying them together.

    Inherits BinaryFilter class.
    """

    def operation(self, element_a, element_b):
        return element_a * element_b


class Division(BinaryFilter):
    """Combines results of the two child filters by dividing the results of the first by the seconds.

    Inherits BinaryFilter class.
    """

    def operation(self, element_a, element_b):
        return element_a / element_b


class IntegerDivision(BinaryFilter):
    """Combines results of the two child filters by perfoming integer division on the results of the first by the results of the second.

    The division is done with python's // operator.

    Inherits BinaryFilter class.
    """

    def operation(self, element_a, element_b):
        return element_a // element_b


class Modulo(BinaryFilter):
    """Combines results of the two child filters by taking the results of the first modulo results of the second.

    Inherits BinaryFilter class.
    """

    def operation(self, element_a, element_b):
        return element_a % element_b


class And(BinaryFilter):
    """Combines results of the two child filters with bitwise AND.

    Inherits BinaryFilter class.
    """

    def operation(self, element_a, element_b):
        return element_a & element_b


class Or(BinaryFilter):
    """Combines results of the two child filters with bitwise OR.

    Inherits BinaryFilter class.
    """

    def operation(self, element_a, element_b):
        return element_a | element_b


class Xor(BinaryFilter):
    """Combines results of the two child filters with bitwise XOR.

    Inherits BinaryFilter class.
    """

    def operation(self, element_a, element_b):
        return element_a ^ element_b


class Difference(Filter):
    """Returns change to data from filter

    Given a filter, applies the filter to the data, then subtracting the original.
    Functions identically to Subtraction(filter, Identity()).

    Inherits BinaryFilter class.
    """

    def __init__(self, ftr):
        super().__init__()
        self.ftr = Subtraction(ftr, Identity())

    def apply(self, node_data, random_state, named_dims):
        self.ftr.apply(node_data, random_state, named_dims)


class Max(BinaryFilter):
    """Combines results of the two child filters by taking the pairwise maximum of the results of the first and second.

    Inherits BinaryFilter class.
    """

    def operation(self, element_a, element_b):
        return max(element_a, element_b)


class Min(BinaryFilter):
    """Combines results of the two child filters by taking the pairwise minimum of the results of the first and second.

    Inherits BinaryFilter class.
    """

    def operation(self, element_a, element_b):
        return min(element_a, element_b)

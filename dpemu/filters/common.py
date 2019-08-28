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

import numpy as np
from dpemu.filters import Filter


class Missing(Filter):
    """Introduces missing values to data.

    For each element in the array, makes that element go missing with the provided probability.
    Values that go missing are replaced with the provided value, which should usually be NaN.

    Inherits Filter class.
    """

    def __init__(self, probability_id, missing_value_id):
        """
        Args:
            probability_id (str): The key mapping to the probability any specific value goes missing.
            missing_value_id (str): The key mapping to the value that represents missing values.
        """
        self.probability_id = probability_id
        self.missing_value_id = missing_value_id
        super().__init__()

    def apply(self, node_data, random_state, named_dims):
        mask = random_state.rand(*(node_data.shape)) <= self.probability
        node_data[mask] = self.missing_value


class Clip(Filter):
    """Clips values between minimum and maximum values provided by the user.

    Sets values less than the minimum value to min, and values greater than the maximum value to max.

    Inherits Filter class.
    """

    def __init__(self, min_id, max_id):
        """
        Args:
            min_id (str): The key mapping to the minimum value.
            max_id (str): The key mapping to the maximum value.
        """
        self.min_id = min_id
        self.max_id = max_id
        super().__init__()

    def apply(self, node_data, random_state, named_dims):
        np.clip(node_data, self.min, self.max, out=node_data)


class GaussianNoise(Filter):
    """Adds normally distributed noise to data.

    Adds random noise drawn from a Gaussian distribution with the provided mean and standard deviation
    to each element in the array.

    Inherits Filter class.
    """

    def __init__(self, mean_id, std_id):
        """
        Args:
            mean_id (str): The key mapping to the mean of the random noise.
            std_id (str): The key mapping to the standard deviation of the random noise.
        """
        self.mean_id = mean_id
        self.std_id = std_id
        super().__init__()

    def apply(self, node_data, random_state, named_dims):
        node_data += random_state.normal(loc=self.mean, scale=self.std, size=node_data.shape).astype(node_data.dtype)


class GaussianNoiseTimeDependent(Filter):
    """Adds normally distributed noise increasing in intensity with time to the data.

    Adds random noise drawn from a Gaussian distribution with mean and standard deviation
    calculated from the initial mean and standard deviation, the elapsed time, and the
    increase to mean and standard deviation per unit of time.

    Inherits Filter class.
    """

    def __init__(self, mean_id, std_id, mean_increase_id, std_increase_id):
        """
        Args:
            mean_id (str): The key mapping to the initial mean of the random noise.
            std_id (str): The key mapping to the initial standard deviation of the random noise.
            mean_increase_id (str): The key mapping to the increase of the mean of the random noise per unit of time.
            std_increase_id (str): The key mapping to the increase of the standard deviation of the random noise per unit of time.
        """
        self.mean_id = mean_id
        self.std_id = std_id
        self.mean_increase_id = mean_increase_id
        self.std_increase_id = std_increase_id
        super().__init__()

    def apply(self, node_data, random_state, named_dims):
        time = named_dims["time"]
        node_data += random_state.normal(loc=self.mean + self.mean_increase * time,
                                         scale=self.std + self.std_increase * time,
                                         size=node_data.shape)


class StrangeBehaviour(Filter):
    """Emulates strange sensor values due to anomalous conditions around the sensor.

    The function do_strange_behaviour given as a parameter is used to output
    strange sensor values into the data.

    Inherits Filter class.
    """

    def __init__(self, do_strange_behaviour_id):
        """
        Args:
            do_strange_behaviour_id (str): The key mapping to the strange behaviour -function.
        """
        super().__init__()
        self.do_strange_behaviour_id = do_strange_behaviour_id

    def apply(self, node_data, random_state, named_dims):
        for index, _ in np.ndenumerate(node_data):
            node_data[index] = self.do_strange_behaviour(node_data[index], random_state)


# TODO: does this work with the new set_params? Where are parameters of the input tuple set?
class ApplyToTuple(Filter):
    """Applies the given filter to only some index in the data tuple.

    Given a filter and an index as parameters, applies the given filter
    to the given index in the data tuple.

    Inherits Filter class.
    """

    def __init__(self, ftr, tuple_index):
        """
        Args:
            ftr (dpemu.filters.Filter): Filter to apply.
            tuple_index (int): Index of the tuple to apply the filter to.
        """
        super().__init__()
        self.ftr = ftr
        self.tuple_index = tuple_index

    def apply(self, node_data, random_state, named_dims):
        self.ftr.apply(node_data[self.tuple_index], random_state, named_dims)


# TODO: does this work with the new set_params? Where are parameters of the input tuple set?
class ApplyWithProbability(Filter):
    """Applies the input filter to the data with the specified probability.

    Inherits Filter class.
    """

    def __init__(self, ftr, probability_id):
        """
        Args:
            ftr_id (str): The key mapping to the filter.
            probability_id (str): The key mapping to the probability of the filter being applied.
        """
        super().__init__()
        self.ftr = ftr
        self.probability_id = probability_id

    def apply(self, node_data, random_state, named_dims):
        if random_state.rand() < self.probability:
            self.ftr.apply(node_data, random_state, named_dims)


# TODO: does this work with the new set_params? Where are parameters of the input tuple set?
class ModifyAsDataType(Filter):
    """Applies the input filter to the data casted to the specified type.

    First casts the data into the specified type, then applies the filter,
    then returns the data to its original type.

    Inherits Filter class.
    """

    def __init__(self, dtype_id, ftr):
        """
        Args:
            dtype_id (str): The key mapping to the data type to cast the input to.
            ftr (dpemu.filters.Filter): The filter to apply to the casted data.
        """

        super().__init__()
        self.dtype_id = dtype_id
        self.ftr = ftr

    def apply(self, node_data, random_state, named_dims):
        copy = node_data.copy().astype(self.dtype)
        self.ftr.apply(copy, random_state, named_dims)
        copy = copy.astype(node_data.dtype)
        for index, _ in np.ndenumerate(node_data):
            node_data[index] = copy[index]

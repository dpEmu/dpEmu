import numpy as np
from dpemu.filters import Filter


class Missing(Filter):
    """Introduce missing values to data.

    For each element in the array, change the value of the element to nan
    with the provided probability.

    Inherits Filter class.
    """

    def __init__(self, probability_id, missing_value_id):
        """
        Args:
            probability_id (str): A key which maps to a probability.
        """
        self.probability_id = probability_id
        self.missing_value_id = missing_value_id
        super().__init__()

    def apply(self, node_data, random_state, named_dims):
        mask = random_state.rand(*(node_data.shape)) <= self.probability
        node_data[mask] = self.missing_value


class Clip(Filter):
    """Clip values to minimum and maximum value provided by the user.

    Inherits Filter class.
    """

    def __init__(self, min_id, max_id):
        """
        Args:
            min_id (str): A key which maps to a minimum value.
            max_id (str): A key which maps to a maximum value.
        """
        self.min_id = min_id
        self.max_id = max_id
        super().__init__()

    def apply(self, node_data, random_state, named_dims):
        np.clip(node_data, self.min, self.max, out=node_data)


class GaussianNoise(Filter):
    """Add normally distributed noise to data.

    For each element in the array add noise drawn from a Gaussian distribution
    with the provided parameters mean and std (standard deviation).

    Inherits Filter class.
    """

    def __init__(self, mean_id, std_id):
        """
        Args:
            mean_id (str): A key which maps to a mean value.
            std_id (str): A key which maps to a standard deviation value.
        """
        self.mean_id = mean_id
        self.std_id = std_id
        super().__init__()

    def apply(self, node_data, random_state, named_dims):
        node_data += random_state.normal(loc=self.mean, scale=self.std, size=node_data.shape).astype(node_data.dtype)


class GaussianNoiseTimeDependent(Filter):
    """Add time dependent normally distributed noise.

    For each element in the array add noise drawn from a Gaussian distribution
    with the provided parameters mean and std (standard deviation). The mean and
    standard deviation increase with every unit of time by the amount specified
    in the last two parameters.

    Inherits Filter class.
    """

    def __init__(self, mean_id, std_id, mean_increase_id, std_increase_id):
        """
        Args:
            mean_id (str): A key which maps to a mean value.
            std_id (str): A key which maps to a standard deviation value.
            mean_increase_id (str): A key which maps to an increase in mean.
            std_increase_id (str): A key which maps to an increase in standard deviation.
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
    """Emulate strange sensor values due to anomalous conditions around the sensor.

    The function do_strange_behaviour is user defined and outputs strange sensor
    values into the data.

    Inherits Filter class.
    """

    def __init__(self, do_strange_behaviour_id):
        """
        Args:
            do_strange_behaviour_id (str): A key which maps to the strange_behaviour function.
        """
        super().__init__()
        self.do_strange_behaviour_id = do_strange_behaviour_id

    def apply(self, node_data, random_state, named_dims):
        for index, _ in np.ndenumerate(node_data):
            node_data[index] = self.do_strange_behaviour(node_data[index], random_state)


class ApplyToTuple(Filter):
    def __init__(self, ftr, tuple_index):
        super().__init__()
        self.ftr = ftr
        self.tuple_index = tuple_index

    def apply(self, node_data, random_state, named_dims):
        self.ftr.apply(node_data[self.tuple_index], random_state, named_dims)


class ApplyWithProbability(Filter):
    """Apply a filter with the specified probability.

    A filter is applied with the specified probability.
    Inherits Filter class.
    """

    def __init__(self, ftr, probability_id):
        """
        Args:
            ftr_id (str): A key which maps to a filter.
            probability_id (str): A key which maps to the probability of the filter being applied.
        """
        super().__init__()
        self.ftr = ftr
        self.probability_id = probability_id

    def apply(self, node_data, random_state, named_dims):
        if random_state.rand() < self.probability:
            self.ftr.apply(node_data, random_state, named_dims)


class ModifyAsDataType(Filter):
    """[summary]

    [extended_summary]

    Inherits Filter class.
    """

    def __init__(self, dtype_id, ftr_id):
        super().__init__()
        self.dtype_id = dtype_id
        self.ftr_id = ftr_id

    def apply(self, node_data, random_state, named_dims):
        copy = node_data.copy().astype(self.dtype)
        self.ftr.apply(copy, random_state, named_dims)
        copy = copy.astype(node_data.dtype)
        for index, _ in np.ndenumerate(node_data):
            node_data[index] = copy[index]

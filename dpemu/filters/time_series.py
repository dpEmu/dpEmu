import numpy as np
from dpemu.filters import Filter


class Gap(Filter):
    """Introduce gaps to time series data by simulating sensor failure.

    Model the state of a sensor as a Markov chain. The sensor always
    starts in a working state. The sensor has a specific probability
    to stop working and a specific probability to start working.

    Inherits Filter class.
    """

    def __init__(self, prob_break_id, prob_recover_id, missing_value_id):
        """
        Args:
            prob_break_id (str): A key which maps to the probability of the sensor breaking.
            prob_recover_id (str): A key which maps to the probability of the sensor recovering.
            missing_value_id (str): A key which maps to a missing value to be used.
        """
        super().__init__()
        self.prob_break_id = prob_break_id
        self.prob_recover_id = prob_recover_id
        self.missing_value_id = missing_value_id
        self.working = True

    def apply(self, node_data, random_state, named_dims):
        def update_working_state():
            if self.working:
                if random_state.rand() < self.prob_break:
                    self.working = False
            else:
                if random_state.rand() < self.prob_recover:
                    self.working = True

        # random_state.rand(node_data.shape[0], node_data.shape[1])
        for index, _ in np.ndenumerate(node_data):
            update_working_state()
            if not self.working:
                node_data[index] = self.missing_value


class SensorDrift(Filter):
    """Emulate sensor values drifting due to a malfunction in the sensor.

    Magnitude is the linear increase in drift during time period t_i -> t_i+1.

    Inherits Filter class.
    """

    def __init__(self, magnitude_id):
        """
        Args:
            magnitude_id (str): A key which maps to the magnitude value.
        """
        super().__init__()
        self.magnitude_id = magnitude_id

    def apply(self, node_data, random_state, named_dims):
        increases = np.arange(1, node_data.shape[0] + 1) * self.magnitude
        node_data += increases.reshape(node_data.shape)

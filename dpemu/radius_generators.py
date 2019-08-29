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

from abc import ABC, abstractmethod


class RadiusGenerator(ABC):
    """Radius generators are used by some filters for generating radii for their effects.
    """

    def __init__(self):
        pass

    @abstractmethod
    def generate(self, random_state):
        """Generates a single integer to be used as a radius in some of the filters.

        Args:
            random_state (mtrand.RandomState): A random state object to be used in all things related to randomness
                to ensure the repeatability.

        Returns:
            int: An integer describing the generated radius.
        """
        pass


class GaussianRadiusGenerator(RadiusGenerator):
    """GaussianRadiusGenerator generates radii from a normal distribution with given parameters.
    """

    def __init__(self, mean, std):
        """
        Args:
            mean (float): The mean of the normal distribution.
            std (float): The standard deviation of the normal distribution.
        """
        self.mean = mean
        self.std = std

    def generate(self, random_state):
        return max(0, self.mean + round(random_state.normal(scale=self.std)))


class ProbabilityArrayRadiusGenerator(RadiusGenerator):
    """ProbabilityArrayRadiusGenerator generates radii based on the probabilities in the array given as a parameter.
    """

    def __init__(self, probability_array):
        """
        Args:
            probability_array (list): A list where the value of an element describes the probability of using its
                index as a radius.
        """
        self.probability_array = probability_array

    def generate(self, random_state):
        sum_of_probabilities = 1
        for radius, _ in enumerate(self.probability_array):
            if random_state.random_sample() <= self.probability_array[radius] / sum_of_probabilities:
                return radius
            sum_of_probabilities -= self.probability_array[radius]
        return 0  # return 0 if for some reason none of the radii is chosen

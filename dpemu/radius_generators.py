from abc import ABC, abstractmethod


class RadiusGenerator(ABC):
    """[summary]

    [extended_summary]
    """

    def __init__(self):
        pass

    @abstractmethod
    def generate(self, random_state):
        """[summary]

        [extended_summary]

        Args:
            random_state ([type]): [description]

        Returns:
            [type]: [description]
        """
        pass


class GaussianRadiusGenerator(RadiusGenerator):
    """[summary]

    [extended_summary]

    Args:
        RadiusGenerator ([type]): [description]
    """

    def __init__(self, mean, std):
        """
        Args:
            mean ([type]): [description]
            std ([type]): [description]
        """
        self.mean = mean
        self.std = std

    def generate(self, random_state):
        """[summary]

        [extended_summary]

        Args:
            random_state ([type]): [description]

        Returns:
            [type]: [description]
        """
        return max(0, self.mean + round(random_state.normal(scale=self.std)))


class ProbabilityArrayRadiusGenerator(RadiusGenerator):
    """[summary]

    [extended_summary]

    Args:
        RadiusGenerator ([type]): [description]
    """

    def __init__(self, probability_array):
        """
        Args:
            probability_array ([type]): [description]
        """
        self.probability_array = probability_array

    def generate(self, random_state):
        """[summary]

        [extended_summary]

        Args:
            random_state ([type]): [description]

        Returns:
            [type]: [description]
        """
        sum_of_probabilities = 1
        for radius, _ in enumerate(self.probability_array):
            if random_state.random_sample() <= self.probability_array[radius] / sum_of_probabilities:
                return radius
            sum_of_probabilities -= self.probability_array[radius]
        return 0  # return 0 if for some reason none of the radii is chosen

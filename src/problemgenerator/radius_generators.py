class RadiusGenerator:
    def __init__(self):
        pass

    def generate(self, random_state):
        print("generate(random_state) function not implemented")
        return 0


class GaussianRadiusGenerator(RadiusGenerator):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def generate(self, random_state):
        return max(0, self.mean + round(random_state.normal(scale=self.std)))


class ProbabilityArrayRadiusGenerator(RadiusGenerator):
    def __init__(self, probability_array):
        self.probability_array = probability_array

    def generate(self, random_state):
        sum_of_probabilities = 1
        for radius, _ in enumerate(self.probability_array):
            if random_state.random_sample() <= self.probability_array[radius] / sum_of_probabilities:
                return radius
            sum_of_probabilities -= self.probability_array[radius]
        return 0  # return 0 if for some reason none of the radii is chosen

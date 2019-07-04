from src.problemgenerator.copy import Copy
from numpy.random import RandomState

class Root(Copy):

    def __init__(self, child):
        super().__init__(child)

    def parameterize(self, params_dict):
        print(f"parametrizing with params dict: {params_dict}")
        self.set_error_params(params_dict)

    def generate_error(self, data, error_params):
        self.parameterize(error_params)
        return self.process(data, RandomState(42))

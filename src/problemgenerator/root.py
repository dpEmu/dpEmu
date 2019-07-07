from src.problemgenerator.copy import Copy
from numpy.random import RandomState


class Root(Copy):
    """An error generation tree should always have a Root object as its root node.
    The Root node itself functions as the error generator – no separate error
    generator object is needed.
    """


    def __init__(self, child):
        super().__init__(child)

    def parameterize(self, params_dict):
        self.set_error_params(params_dict)

    def generate_error(self, data, error_params):
        """Returns the data with the desired errors introduced. The original
        data object is not modified. The error parameters must be provided as
        a dictionary whose keys are the parameter identifiers (given as
        parameters to the filters) and whose values are the desired parameter
        values.
        """

        self.parameterize(error_params)
        return self.process(data, RandomState(42))

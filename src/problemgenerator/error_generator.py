import numpy as np
from src.problemgenerator.copy import Copy

class ErrGen:
    def __init__(self, root_node):
        self.random_state = np.random.RandomState(42)
        self.root_node = Copy(root_node)

    def generate_error(self, data, error_params):
        return self.root_node.process(data, self.random_state)

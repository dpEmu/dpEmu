import numpy as np


class Series:

    def __init__(self, child):
        self.child = child

    def process(self, data):
        data_length = data.shape[0]
        return np.array([self.child.process(data[i, ...]) for i in range(data_length)])


class TupleSeries:

    def __init__(self, children):
        self.children = children

    def process(self, data):
        data_length = data[0].shape[0]
        as_list = [np.array(
            [child.process(data[index][j, ...]) for j in range(data_length)])
            for (index, child) in enumerate(self.children)]
        return tuple(as_list)

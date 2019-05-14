import numpy as np

class Series:

  def __init__(self, child):
    self.child = child

  def process(self, data):
    data_length = data.shape[0]
    return np.array([self.child.process(data[i,...]) for i in range(data_length)])
  
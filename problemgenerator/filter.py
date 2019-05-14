import numpy as np

class Filter:
  
  def __init__(self):
    self.shape = ()

class Missing(Filter):

  def __init__(self, probability):
    self.probability = probability
    super().__init__()

  def apply(self, data):
    mask = np.random.choice([True, False],
      size=self.shape,
      p=[self.probability, 1. - self.probability])
    copy = data.copy()
    copy[mask] = np.nan
    return copy
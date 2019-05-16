import numpy as np

class Filter:
  
  def __init__(self):
    pass

class Missing(Filter):

  def __init__(self, probability):
    self.probability = probability
    super().__init__()

  def apply(self, data):
    mask = np.random.choice([True, False],
      size=data.shape,
      p=[self.probability, 1. - self.probability])
    copy = data.copy()
    copy[mask] = np.nan
    return copy
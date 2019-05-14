class Tensor:

  def __init__(self, shape):
    self.shape = shape
    self.filters = []

  def addfilter(self, filter):
    self.filters.append(filter)
    filter.shape = self.shape

  def process(self, input):
    output = input
    for filter in self.filters:
      output = filter.apply(output)
    return output
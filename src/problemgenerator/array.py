class Array:

    def __init__(self, shape):
        self.shape = shape
        self.filters = []

    def addfilter(self, custom_filter):
        self.filters.append(custom_filter)
        custom_filter.shape = self.shape

    def process(self, input_source):
        output = input_source
        for f in self.filters:
            output = f.apply(output)
        return output

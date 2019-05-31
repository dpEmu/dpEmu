class Series:

    def __init__(self, child):
        self.child = child

    def process(self, data):
        data_length = data.shape[0]
        for i in range(data_length):
            self.child.process(data[i, ...])

class TupleSeries:

    def __init__(self, children):
        self.children = children

    def process(self, data):
        data_length = data[0].shape[0]
        for i, child in enumerate(self.children):
            for j in range(data_length):
                child.process(data[i][j, ...])

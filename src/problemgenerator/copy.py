import copy

class Copy:
    def __init__(self, child):
        self.child = child

    def process(self, data):
        copy_data = copy.deepcopy(data)
        self.child.process(copy_data)
        return copy_data

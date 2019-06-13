import copy


class Copy:
    def __init__(self, child):
        self.child = child

    def process(self, data, random_state):
        copy_data = copy.deepcopy(data)
        self.child.process(copy_data, random_state)
        return copy_data

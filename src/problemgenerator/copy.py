from src.problemgenerator.node import Node
import copy


class Copy(Node):
    def __init__(self, child):
        super().__init__([child])

    def process(self, data, random_state):
        copy_data = copy.deepcopy(data)
        self.children[0].process(copy_data, random_state)
        return copy_data

from copy import deepcopy

import numpy as np

from src import runner


# Model: ax+b = y, find a, minimize \sum |ax+b - y|
class Model:
    def __init__(self, b):
        self.b = b

    def run(self, data, model_params):
        a = model_params["a"]
        b = self.b
        err = 0
        for (x, y) in data:
            err += abs(a * x + b - y)
        return {"error": err}


# Add gaussian noise with parameters to data
class ErrGen:
    def __init__(self, data, seed):
        self.data = data
        self.seed = seed

    def generate_error(self, params):
        res = deepcopy(self.data)
        np.random.seed(self.seed)
        for (x, _) in res:
            x += np.random.normal(loc=params["mean"], scale=params["std"])
        return res


# Ternary searches best parameter
class ParamSelector:
    def __init__(self, low, high, eps, params):
        self.low = low
        self.high = high
        self.eps = eps
        self.params = params

    def next(self):
        if self.high - self.low <= self.eps:
            return None
        left_mid = (2 * self.low + self.high) / 3
        right_mid = (self.low + 2 * self.high) / 3
        return [(self.params, {"a": a}) for a in [left_mid, right_mid]]

    def analyze(self, res):
        left_mid = (2 * self.low + self.high) / 3
        right_mid = (self.low + 2 * self.high) / 3
        if res[0]["error"] <= res[1]["error"]:
            self.high = right_mid
        else:
            self.low = left_mid


# Example usage
def main():
    data = [(3, 6), (0, 1), (5, 9), (10, 22)]

    model = Model(1)
    err_gen = ErrGen(data, 0)
    param_selector = ParamSelector(0, 10, 0.001, {"mean": 0, "std": 1})

    res = runner.run(model, err_gen, param_selector)
    print(res)


# Call main
main()

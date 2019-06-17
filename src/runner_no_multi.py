import time
from copy import deepcopy

import pandas as pd

def run(model, errgen, param_chooser):
    """ Runs the model (NOT in parallel!) while parameters for error generation are provided.
        This is a temporary kludge based on the parallel implementation.
        errgen: class for generating erronous data. Should have function
            .generate_error(err_param), that adds error to the data according to the parameters.
        param_chooser: class for selecting parameters for error generation and the model.
            .next() should return an array of pairs (err_param, mod_param), where
            err_param is the parameters for error generation, and mod_param is the parameters for the model.
            .analyze(outputs) is given an array of the outputs of the runs of the model with the last parameters.
            this function can modify param_chooser's inner state so better next parameters can be selected
        model: class for the ML model. Can be pretrained or be trained when .run is called. Should have function
            .run(data, model_params=...) where data is the data the model is run on,
            and model_params is the parameters returned by param_chooser. it should return an dictionary
            describing the results in some way.
    """

    rows = []
    batches = 0
    while True:
        batches += 1
        params = param_chooser.next()
        if not params:
            break

        pool_input = []
        for i, param_pair in enumerate(params):
            (err_param, mod_param) = param_pair
            mod_data = errgen.generate_error(err_param)
            pool_input.append((deepcopy(model), mod_data, mod_param))
        outputs = [mod.run(data, model_params=mp) for (mod, data, mp) in pool_input]

        param_chooser.analyze(outputs)
        for i, param_pair in enumerate(params):
            (outputs[i]["err_param"], outputs[i]["mod_param"]) = param_pair
            outputs[i]["batch"] = batches
            rows.append(outputs[i])
    return pd.DataFrame(rows)

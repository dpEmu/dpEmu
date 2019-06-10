from threading import Thread
from copy import deepcopy
import pandas as pd

def run(model, errgen, param_chooser):
    ''' Runs the model in parallel while parameters for error generation are provided.
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
    '''
    def run_model(mod, out, i, md, mp):
        out[i] = mod.run(md, model_params=mp)

    rows = []
    batches = 0
    while(True):
        batches += 1
        params = param_chooser.next()
        if not params:
            break

        processes = []
        outputs = [None] * len(params)

        for i, param_pair in enumerate(params):
            (err_param, mod_param) = param_pair
            mod_data = errgen.generate_error(err_param)
            p = Thread(target=run_model,
                       args=(deepcopy(model), outputs, i, mod_data, mod_param))
            p.start()
            processes.append(p)

        for i, p in enumerate(processes):
            p.join()

        param_chooser.analyze(outputs)

        for i, p in enumerate(processes):
            (outputs[i]["err_param"], outputs[i]["mod_param"]) = params[i]
            outputs[i]["batch"] = batches
            rows.append(outputs[i])
    return pd.DataFrame(rows)

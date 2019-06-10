from multiprocessing import Process
from copy import deepcopy
from inspect import getfullargspec
import pandas as pd

def runner(data, conf, errgen, param_chooser, model):
    ''' Runs the model in parallel while parameters for error generation are provided.
        data: original data
        conf: constant global config for error generation, param_chooser and model
        errgen: function for generating erronous data. Should have signature errgen(data, param, conf)
        param_chooser: class for selecting parameters for error generation and the model.
            param_chooser.next(conf) should return an array of pairs (err_param, mod_param), where
            err_param is the parameters for error generation, and mod_param is the parameters for the model.
        model: class for the ML model. Can be pretrained or be trained when .run is called. Should have function
            model.run(data, conf) or model.run(data, conf, model_params=...) where data is the data the model is run on,
            conf is the config given to this function, and model_params is the parameters returned by param_chooser.
            it should return an dictionary describing the results in some way.
    '''
    def run_model(mod, out, i, md, mp, c):
        spec = getfullargspec(mod.run)
        if "model_params" in spec.args:
            out[i] = mod.run(md, c, model_params=mp)
        else:
            out[i] = mod.run(md, c)

    rows = []
    batches = 0
    while(True):
        batches += 1
        params = param_chooser.next(conf)
        if not params:
            break
        processes = []
        outputs = []
        for i, param_pair in enumerate(params):
            (err_param, mod_param) = param_pair
            mod_data = errgen(data, err_param, conf)
            p = Process(target=run_model,
                        args=(deepcopy(model), outputs, i, mod_data, mod_param, deepcopy(conf)))
            p.start()
            processes.append(p)
        for i, p in enumerate(processes):
            p.join()
            (outputs[i]["err_param"], outputs[i]["mod_param"]) = params[i]
            outputs[i]["batch"] = batches
            rows.append(outputs[i])
    return pd.dataframe(rows)

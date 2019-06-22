import multiprocessing
import time

import pandas as pd


def worker(inputs):
    mod, mp, ep, ed = inputs
    start_time = time.time()
    if mp:
        res = mod.run(ed, model_params=mp)
    else:
        res = mod.run(ed)
    time_used = time.time() - start_time
    res.update({k: v for k, v in ep.items()})
    res.update({k: v for k, v in mp.items()})
    res["time_used"] = time_used
    return res


def run(errgen, err_params_list, model_param_pairs):
    results = {}
    for err_params in err_params_list:
        err_data = errgen.generate_error(err_params)
        pool_inputs = []
        for model_param_pair in model_param_pairs:
            model = model_param_pair[0]
            for model_params in model_param_pair[1]:
                pool_inputs.append((model, model_params, err_params, err_data))
        with multiprocessing.Pool() as pool:
            outputs = pool.map(worker, pool_inputs)
        for i, pool_input in enumerate(pool_inputs):
            model_name = pool_input[0].__class__.__name__.replace("Model", "")
            if model_name in results:
                results[model_name].append(outputs[i])
            else:
                results[model_name] = [outputs[i]]
    dfs = []
    for model_name, result in results.items():
        df = pd.DataFrame(result)
        df.name = model_name
        dfs.append(df)
    return dfs

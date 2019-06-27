import multiprocessing
import time

import pandas as pd
from tqdm import tqdm


def worker(inputs):
    model_param_pairs, err_gen, err_params = inputs
    time_start = time.time()
    err_data = err_gen.generate_error(err_params)
    time_used_err = time.time() - time_start
    results = []
    for model_param_pair in tqdm(model_param_pairs):
        model, model_params_list = model_param_pair
        model_name = model.__name__.replace("Model", "")
        for model_params in model_params_list:
            time_start = time.time()
            result = model().run(err_data, model_params)
            time_used_mod = time.time() - time_start
            result["model_name"] = model_name
            result["time_used_err"] = time_used_err
            result["time_used_mod"] = time_used_mod
            result.update({k: v for k, v in err_params.items()})
            result.update({k: v for k, v in model_params.items()})
            results.append(result)
    return results


def run(err_gen, err_params_list, model_param_pairs):
    pool_inputs = []
    for err_params in err_params_list:
        pool_inputs.append((model_param_pairs, err_gen, err_params))
    with multiprocessing.Pool() as pool:
        outputs = pool.map(worker, pool_inputs)
    rows = [result for results in outputs for result in results]
    return pd.DataFrame(rows)

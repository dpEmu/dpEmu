import multiprocessing
import time

import pandas as pd
from tqdm import tqdm


def worker(inputs):
    model_param_pairs, err_gen, err_params = inputs
    err_data = err_gen.generate_error(err_params)
    results = []
    for model_param_pair in tqdm(model_param_pairs):
        model, model_params_list = model_param_pair
        model_name = model.__name__.replace("Model", "")
        for model_params in model_params_list:
            start_time = time.time()
            result = model().run(err_data, model_params)
            time_used = time.time() - start_time
            result["time_used"] = time_used
            result["model_name"] = model_name
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
    df = pd.DataFrame(rows)
    dfs = []
    for model_name, df_ in df.groupby("model_name"):
        df_ = df_.dropna(axis=1, how="all")
        df_ = df_.drop("model_name", axis=1)
        df_ = df_.reset_index(drop=True)
        df_.name = model_name
        dfs.append(df_)
    return dfs

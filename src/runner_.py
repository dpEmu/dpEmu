import time
from multiprocessing.pool import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm


def worker(inputs):
    train, test, preproc, err_gen, err_params, mod_params_dict_list, use_i_mode = inputs

    time_start = time.time()
    err_train = None
    if train is not None:
        train = np.array(train)
        err_train = err_gen().generate_error(train, err_params)
    test = np.array(test)
    err_test = err_gen().generate_error(test, err_params)
    time_used_err = time.time() - time_start

    preproc_train, preproc_err_test_using_train, res_base_using_train = preproc().run(train, err_test)
    preproc_err_train, preproc_err_test_using_err_train, res_base_using_err_train = preproc().run(err_train, err_test)

    results = []
    for mod_params_dict in mod_params_dict_list:
        mod = mod_params_dict["model"]
        mod_params_list = mod_params_dict["params_list"]
        if "use_clean_train_data" in mod_params_dict:
            use_clean_train = mod_params_dict["use_clean_train_data"]
        else:
            use_clean_train = False

        if use_clean_train:
            mod_name = mod.__name__.replace("Model", "Clean")
        else:
            mod_name = mod.__name__.replace("Model", "")

        for mod_params in mod_params_list:
            time_start = time.time()
            if use_clean_train:
                res = mod().run(preproc_train, preproc_err_test_using_train, mod_params)
                res.update(res_base_using_train)
            else:
                res = mod().run(preproc_err_train, preproc_err_test_using_err_train, mod_params)
                res.update(res_base_using_err_train)
            time_used_mod = time.time() - time_start

            if use_i_mode:
                res["interactive_err_data"] = err_test
            res["model_name"] = mod_name
            res["time_used_err"] = time_used_err
            res["time_used_mod"] = time_used_mod
            res.update({k: v for k, v in err_params.items()})
            res.update({k: v for k, v in mod_params.items()})

            results.append(res)
    return results


def run(train, test, preproc, err_gen, err_params_list, mod_params_dict_list, use_i_mode=False):
    inputs = []
    for err_params in err_params_list:
        inputs.append((train, test, preproc, err_gen, err_params, mod_params_dict_list, use_i_mode))
    total_results = []
    with Pool() as pool:
        for results in tqdm(pool.imap(worker, inputs), total=len(err_params_list)):
            total_results.extend(results)
    return pd.DataFrame(total_results)

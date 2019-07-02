import multiprocessing
import time

import pandas as pd


def worker(inputs):
    train_data, test_data, err_gen, err_params, model_params_tuple_list, use_interactive_mode = inputs

    time_start = time.time()
    if train_data:
        err_train_data = err_gen().generate_error(train_data, err_params)
    else:
        err_train_data = None
    err_test_data = err_gen().generate_error(test_data, err_params)
    time_used_err = time.time() - time_start

    results = []
    for model_params_tuple in model_params_tuple_list:
        if len(model_params_tuple) == 3:
            model, model_params_list, use_clean_train_data = model_params_tuple
        else:
            model, model_params_list = model_params_tuple
            use_clean_train_data = False

        if use_clean_train_data:
            model_name = model.__name__.replace("Model", "Clean")
        else:
            model_name = model.__name__.replace("Model", "")

        for model_params in model_params_list:
            time_start = time.time()
            if use_clean_train_data:
                result = model().run(train_data, err_test_data, model_params)
            else:
                result = model().run(err_train_data, err_test_data, model_params)
            time_used_mod = time.time() - time_start

            if use_interactive_mode:
                result["err_test_data"] = err_test_data
            result["model_name"] = model_name
            result["time_used_err"] = time_used_err
            result["time_used_mod"] = time_used_mod
            result.update({k: v for k, v in err_params.items()})
            result.update({k: v for k, v in model_params.items()})

            results.append(result)
    return results


def run(train_data, test_data, err_gen, err_params_list, model_params_tuple_list, use_interactive_mode=False):
    pool_inputs = []
    for err_params in err_params_list:
        pool_inputs.append((train_data, test_data, err_gen, err_params, model_params_tuple_list, use_interactive_mode))
    with multiprocessing.Pool() as pool:
        outputs = pool.map(worker, pool_inputs)
    rows = [result for results in outputs for result in results]
    return pd.DataFrame(rows)

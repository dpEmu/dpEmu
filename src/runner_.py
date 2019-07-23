import time
from multiprocessing.pool import Pool

import pandas as pd
from tqdm import tqdm


def worker(inputs):
    train_data, test_data, preproc, err_root_node, err_params, model_params_dict_list, use_interactive_mode = inputs
    time_start = time.time()
    err_train_data = None
    if train_data:
        err_train_data = err_root_node.generate_error(train_data, err_params)
    err_test_data = err_root_node.generate_error(test_data, err_params)
    time_used_err = time.time() - time_start

    time_start = time.time()
    preproc_train_data, preproc_err_test_using_train, result_base_using_train = preproc().run(train_data, err_test_data)
    preproc_err_train_data, preproc_err_test_using_err_train, result_base_using_err_train = preproc().run(
        err_train_data, err_test_data)
    time_used_preproc = time.time() - time_start

    results = []
    for model_params_dict in model_params_dict_list:
        model = model_params_dict["model"]
        model_params_list = model_params_dict["params_list"]
        if "use_clean_train_data" in model_params_dict:
            use_clean_train_data = model_params_dict["use_clean_train_data"]
        else:
            use_clean_train_data = False

        if use_clean_train_data:
            model_name = model.__name__.replace("Model", "Clean")
        else:
            model_name = model.__name__.replace("Model", "")

        for model_params in model_params_list:
            time_start = time.time()
            if use_clean_train_data:
                result = model().run(preproc_train_data, preproc_err_test_using_train, model_params)
                result.update(result_base_using_train)
            else:
                result = model().run(preproc_err_train_data, preproc_err_test_using_err_train, model_params)
                result.update(result_base_using_err_train)
            time_used_mod = time.time() - time_start

            if use_interactive_mode:
                result["interactive_err_data"] = err_test_data
            result["model_name"] = model_name
            result["time_used_err"] = time_used_err
            result["time_used_preproc"] = time_used_preproc
            result["time_used_mod"] = time_used_mod
            result.update({k: v for k, v in err_params.items()})
            result.update({k: v for k, v in model_params.items()})

            results.append(result)
    return results


def run(train_data, test_data, preproc, err_root_node, err_params_list, model_params_dict_list, n_processes=None,
        use_interactive_mode=False):
    pool_inputs = [(
        train_data,
        test_data,
        preproc,
        err_root_node,
        err_params,
        model_params_dict_list,
        use_interactive_mode
    ) for err_params in err_params_list]
    total_results = []
    with Pool(n_processes) as pool:
        for results in tqdm(pool.imap(worker, pool_inputs), total=len(err_params_list)):
            total_results.extend(results)
    return pd.DataFrame(total_results)

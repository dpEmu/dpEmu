# MIT License
#
# Copyright (c) 2019 Tuomas Halvari, Juha Harviainen, Juha Mylläri, Antti Röyskö, Juuso Silvennoinen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import time
from collections import Counter
from multiprocessing.pool import Pool
from pickle import dump, load

import pandas as pd
from tqdm import tqdm

from dpemu.utils import generate_unique_path


def _unpickle_data(path_to_train_data, path_to_test_data):
    with open(path_to_train_data, "rb") as file:
        train_data = load(file)
    with open(path_to_test_data, "rb") as file:
        test_data = load(file)
    return train_data, test_data


def _errorify_data(train_data, test_data, err_root_node, err_params):
    time_start = time.time()
    if train_data is not None:
        err_train_data = err_root_node.generate_error(train_data, err_params)
    else:
        err_train_data = None
    err_test_data = err_root_node.generate_error(test_data, err_params)
    time_err = time.time() - time_start
    return err_train_data, err_test_data, time_err


def _preproc_data(train_data, err_train_data, err_test_data, preproc, preproc_params):
    time_start = time.time()
    preproc_train_data, preproc_err_test_using_train, result_base_using_train = preproc().run(
        train_data, err_test_data, preproc_params)
    preproc_err_train_data, preproc_err_test_using_err_train, result_base_using_err_train = preproc().run(
        err_train_data, err_test_data, preproc_params)
    time_pre = time.time() - time_start
    return (
        preproc_train_data, preproc_err_test_using_train, result_base_using_train, preproc_err_train_data,
        preproc_err_test_using_err_train, result_base_using_err_train, time_pre
    )


def _get_model_name(model, use_clean_train_data, same_model_counter):
    model_name = model.__name__.replace("Model", "")
    if use_clean_train_data:
        model_name += "Clean"
    same_model_counter[model_name] += 1
    return model_name + f" #{same_model_counter[model_name]}"


def _get_result_with_model_params(model, model_params, train_data, test_data, result_base):
    time_start = time.time()
    result = model().run(train_data, test_data, model_params)
    result.update(result_base)
    time_mod = time.time() - time_start
    result["time_mod"] = round(time_mod, 3)
    result.update({k: v for k, v in model_params.items()})
    return result


def _get_results_from_model(model, model_params_list, train_data, test_data, result_base):
    if not model_params_list:
        model_params_list.append({})
    return [
        _get_result_with_model_params(model, model_params, train_data, test_data, result_base) for model_params in
        model_params_list
    ]


def _add_more_stuff_to_results(result, err_params, model_name, i_data, time_pre, time_err, use_i_mode):
    result.update({k: v for k, v in err_params.items()})
    result["model_name"] = model_name
    if use_i_mode:
        result["interactive_err_data"] = i_data
    result["time_err"] = round(time_err, 3)
    result["time_pre"] = round(time_pre, 3)


def worker(inputs):
    """[summary]

    [extended_summary]

    Args:
        inputs ([type]): [description]

    Returns:
        [type]: [description]
    """
    (
        path_to_train_data, path_to_test_data, preproc, preproc_params, err_root_node, err_params,
        model_params_dict_list, use_interactive_mode
    ) = inputs
    train_data, test_data = _unpickle_data(path_to_train_data, path_to_test_data)

    err_train_data, err_test_data, time_err = _errorify_data(train_data, test_data, err_root_node, err_params)

    (
        preproc_train_data, preproc_err_test_using_train, result_base_using_train, preproc_err_train_data,
        preproc_err_test_using_err_train, result_base_using_err_train, time_pre
    ) = _preproc_data(train_data, err_train_data, err_test_data, preproc, preproc_params)

    worker_results = []
    same_model_counter = Counter()
    for model_params_dict in model_params_dict_list:
        if "use_clean_train_data" in model_params_dict:
            use_clean_train_data = model_params_dict["use_clean_train_data"]
        else:
            use_clean_train_data = False
        model = model_params_dict["model"]
        model_params_list = model_params_dict["params_list"]
        model_name = _get_model_name(model, use_clean_train_data, same_model_counter)

        if use_clean_train_data:
            results = _get_results_from_model(
                model, model_params_list, preproc_train_data, preproc_err_test_using_train, result_base_using_train
            )
        else:
            results = _get_results_from_model(
                model, model_params_list, preproc_err_train_data, preproc_err_test_using_err_train,
                result_base_using_err_train
            )
        for result in results:
            _add_more_stuff_to_results(result, err_params, model_name, err_test_data, time_pre, time_err,
                                       use_interactive_mode)
        worker_results.extend(results)
    return worker_results


def _pickle_data(train_data, test_data):
    path_to_train_data = generate_unique_path("tmp", "p")
    path_to_test_data = generate_unique_path("tmp", "p")
    with open(path_to_train_data, "wb") as file:
        dump(train_data, file)
    with open(path_to_test_data, "wb") as file:
        dump(test_data, file)
    return path_to_train_data, path_to_test_data


def _get_total_results_from_workers(pool_inputs, n_err_params, n_processes):
    total_results = []
    with Pool(n_processes) as pool:
        for results in tqdm(pool.imap(worker, pool_inputs), total=n_err_params):
            total_results.extend(results)
    return total_results


def get_df_columns_base(err_params_list, model_params_dict_list):
    err_param_columns = set()
    model_param_columns = set()
    [err_param_columns.add(k) for err_params in err_params_list for k, _ in err_params.items()]
    err_param_columns = sorted(err_param_columns)
    [model_param_columns.add(k) for model_params_dict in model_params_dict_list for params_list in
     model_params_dict["params_list"] for k, _ in params_list.items()]
    model_param_columns = sorted(model_param_columns)
    return err_param_columns + model_param_columns + ["time_err", "time_pre", "time_mod"]


def order_df_columns(df, err_params_list, model_params_dict_list):
    df_columns_base = get_df_columns_base(err_params_list, model_params_dict_list)
    new_columns = [column for column in df.columns if column not in df_columns_base]
    return df.reindex(columns=new_columns + df_columns_base)


def run(train_data, test_data, preproc, preproc_params, err_root_node, err_params_list, model_params_dict_list,
        n_processes=None, use_interactive_mode=False):
    """[summary]

    [extended_summary]

    Args:
        train_data ([type]): [description]
        test_data ([type]): [description]
        preproc ([type]): [description]
        err_root_node ([type]): [description]
        err_params_list ([type]): [description]
        model_params_dict_list ([type]): [description]
        n_processes ([type], optional): [description]. Defaults to None.
        use_interactive_mode (bool, optional): [description]. Defaults to False.
    """
    path_to_train_data, path_to_test_data = _pickle_data(train_data, test_data)
    pool_inputs = [(
        path_to_train_data,
        path_to_test_data,
        preproc,
        preproc_params,
        err_root_node,
        err_params,
        model_params_dict_list,
        use_interactive_mode
    ) for err_params in err_params_list]

    total_results = _get_total_results_from_workers(pool_inputs, len(err_params_list), n_processes)
    df = pd.DataFrame(total_results)
    return order_df_columns(df, err_params_list, model_params_dict_list)

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


def unpickle_data(path_to_train_data, path_to_test_data):
    """Loads the data to memory in a subprocess.

    Args:
        path_to_train_data: Path to the train data.
        path_to_test_data: Path to the test data.

    Returns:
        The train and test data.
    """
    with open(path_to_train_data, "rb") as file:
        train_data = load(file)
    with open(path_to_test_data, "rb") as file:
        test_data = load(file)
    return train_data, test_data


def errorify_data(train_data, test_data, err_root_node, err_params):
    """Applies the error to the data using the error source defined.

    Args:
        train_data: The train data.
        test_data: The test data.
        err_root_node: Error root node.
        err_params: Error parameters.

    Returns:
        Erroneous data and time used in error generation.
    """
    time_start = time.time()
    if train_data is not None:
        err_train_data = err_root_node.generate_error(train_data, err_params)
    else:
        err_train_data = None
    err_test_data = err_root_node.generate_error(test_data, err_params)
    time_err = time.time() - time_start
    return err_train_data, err_test_data, time_err


def preproc_data(train_data, err_train_data, err_test_data, preproc, preproc_params):
    """
    Preprocesses clean train data, errorified train data and errorified test data using the given preprocessor and
    parameters.

    Args:
        train_data: The train data.
        err_train_data: Errorified train data.
        err_test_data: Errorified test data.
        preproc: The preprocessor class.
        preproc_params: The preprocessor parameters.

    Returns:
        Preprocessed clean train data, preprocessed errorified test data using clean train data, result dict base when
        using clean train data, preprocessed errorified train data, preprocessed errorified test data when using
        errorified train data, result dict base when using errorified traindata and time used in preprocessing.
    """
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


def get_model_name(model, use_clean_train_data, same_model_counter):
    """
    Returns the name of the model class. If the name ends with the word Model, it's replaced with an empty string. If
    clean train data is used, word Clean is added to the name. A number is added to the end to separate multiple same
    models.

    Args:
        model: The ML model class used.
        use_clean_train_data: True if clean train data is used.
        same_model_counter: Counter used to separate same models.

    Returns:
        The model name.
    """
    model_name = model.__name__.replace("Model", "")
    if use_clean_train_data:
        model_name += "Clean"
    same_model_counter[model_name] += 1
    return model_name + f" #{same_model_counter[model_name]}"


def get_result_with_model_params(model, model_params, train_data, test_data, result_base):
    """Gets the results from a model using specified model parameters.

    Args:
        model: The ML model class used.
        model_params: The model parameters used.
        train_data: The train data.
        test_data: The test data.
        result_base: Base results from the preprocessor.

    Returns:
        The results in a dict.
    """
    time_start = time.time()
    result = model().run(train_data, test_data, model_params)
    result.update(result_base)
    time_mod = time.time() - time_start
    result["time_mod"] = round(time_mod, 3)
    result.update({k: v for k, v in model_params.items()})
    return result


def get_results_from_model(model, model_params_list, train_data, test_data, result_base):
    """Gets all results from a model using different hyperparameter combinations.

    Args:
        model: The ML model class used.
        model_params_list: A list of different hyperparameter combinations for this model.
        train_data: The train data.
        test_data: The test data.
        result_base: Base results from the preprocessor.

    Returns:
        A list of result dicts from the model.
    """
    if not model_params_list:
        model_params_list.append({})
    return [
        get_result_with_model_params(model, model_params, train_data, test_data, result_base) for model_params in
        model_params_list
    ]


def add_more_stuff_to_results(result, err_params, model_name, i_data, time_pre, time_err, use_i_mode):
    """Adds stuff like error parameters, model parameters and run times to a result dict.

    Args:
        result: A result dict.
        err_params: Error parameters.
        model_name: Name of the model.
        i_data: The interactive data.
        time_pre: Time used in the preprocessing phase.
        time_err: Time used in the error generation phase.
        use_i_mode: True if interactive mode is used.
    """
    result.update({k: v for k, v in err_params.items()})
    result["model_name"] = model_name
    if use_i_mode:
        result["interactive_err_data"] = i_data
    result["time_err"] = round(time_err, 3)
    result["time_pre"] = round(time_pre, 3)


def worker(inputs):
    """
    One of the workers in the multiprocessing pool. A subprocess is created for every error parameter combination. In
    every worker, data is first errorified, preprocessed and then run through the models.

    Args:
        inputs: Tuple containing the worker inputs.

    Returns:
        List of all result dicts from different models.
    """
    (
        path_to_train_data, path_to_test_data, preproc, preproc_params, err_root_node, err_params,
        model_params_dict_list, use_interactive_mode
    ) = inputs
    train_data, test_data = unpickle_data(path_to_train_data, path_to_test_data)

    err_train_data, err_test_data, time_err = errorify_data(train_data, test_data, err_root_node, err_params)

    (
        preproc_train_data, preproc_err_test_using_train, result_base_using_train, preproc_err_train_data,
        preproc_err_test_using_err_train, result_base_using_err_train, time_pre
    ) = preproc_data(train_data, err_train_data, err_test_data, preproc, preproc_params)

    worker_results = []
    same_model_counter = Counter()
    for model_params_dict in model_params_dict_list:
        if "use_clean_train_data" in model_params_dict:
            use_clean_train_data = model_params_dict["use_clean_train_data"]
        else:
            use_clean_train_data = False
        model = model_params_dict["model"]
        model_params_list = model_params_dict["params_list"]
        model_name = get_model_name(model, use_clean_train_data, same_model_counter)

        if use_clean_train_data:
            results = get_results_from_model(
                model, model_params_list, preproc_train_data, preproc_err_test_using_train, result_base_using_train
            )
        else:
            results = get_results_from_model(
                model, model_params_list, preproc_err_train_data, preproc_err_test_using_err_train,
                result_base_using_err_train
            )
        for result in results:
            add_more_stuff_to_results(result, err_params, model_name, err_test_data, time_pre, time_err,
                                      use_interactive_mode)
        worker_results.extend(results)
    return worker_results


def pickle_data(train_data, test_data):
    """Saves the data to disk to be read by the workers.

    Args:
        train_data: The train data.
        test_data: The test data.

    Returns:
        Paths to train and test data.
    """
    path_to_train_data = generate_unique_path("tmp", "p")
    path_to_test_data = generate_unique_path("tmp", "p")
    with open(path_to_train_data, "wb") as file:
        dump(train_data, file)
    with open(path_to_test_data, "wb") as file:
        dump(test_data, file)
    return path_to_train_data, path_to_test_data


def get_total_results_from_workers(pool_inputs, n_err_params, n_processes):
    """Gathers the results from different workers to a list.

    Args:
        pool_inputs: List of inputs for different workers.
        n_err_params: Number off error parameter combinations.
        n_processes: Max number of active subprocesses.

    Returns:
        List of all result dicts from different workers.
    """
    total_results = []
    with Pool(n_processes) as pool:
        for results in tqdm(pool.imap(worker, pool_inputs), total=n_err_params):
            total_results.extend(results)
    return total_results


def get_df_columns_base(err_params_list, model_params_dict_list):
    """Generates the base for a list of Dataframe column names.

    Args:
        err_params_list: List of all error parameter combinations.
        model_params_dict_list: List of dicts where each dict includes the class of the model and a list of different
            hyperparameter combinations.

    Returns:
        Base list for Dataframe column names.
    """
    err_param_columns = set()
    model_param_columns = set()
    [err_param_columns.add(k) for err_params in err_params_list for k, _ in err_params.items()]
    err_param_columns = sorted(err_param_columns)
    [model_param_columns.add(k) for model_params_dict in model_params_dict_list for params_list in
     model_params_dict["params_list"] for k, _ in params_list.items()]
    model_param_columns = sorted(model_param_columns)
    return err_param_columns + model_param_columns + ["time_err", "time_pre", "time_mod"]


def order_df_columns(df, err_params_list, model_params_dict_list):
    """Defines the final order for Dataframe column names.

    Args:
        df: A Dataframe containing the results.
        err_params_list: List of all error parameter combinations.
        model_params_dict_list: List of dicts where each dict includes the class of the model and a list of different
            hyperparameter combinations.

    Returns:
        The reindexed Dataframe.
    """
    df_columns_base = get_df_columns_base(err_params_list, model_params_dict_list)
    new_columns = [column for column in df.columns if column not in df_columns_base]
    return df.reindex(columns=new_columns + df_columns_base)


def run(train_data, test_data, preproc, preproc_params, err_root_node, err_params_list, model_params_dict_list,
        n_processes=None, use_interactive_mode=False):
    """
    The runner system is called with the run function. It creates a Pandas Dataframe from all of the results it gets
    from different workers.

    Args:
        train_data: The train data.
        test_data: The test data.
        preproc: The preprocessor class.
        preproc_params: The preprocessor parameters.
        err_root_node: Error root node.
        err_params_list: List of all error parameter combinations.
        model_params_dict_list: List of dicts where each dict includes the class of the model and a list of different
            hyperparameter combinations.
        n_processes: Max number of active subprocesses.
        use_interactive_mode: True if interactive mode is used. The resulting Dataframe contains the errorified data.

    Returns:
        A Dataframe containing the results.
    """
    path_to_train_data, path_to_test_data = pickle_data(train_data, test_data)
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

    total_results = get_total_results_from_workers(pool_inputs, len(err_params_list), n_processes)
    df = pd.DataFrame(total_results)
    return order_df_columns(df, err_params_list, model_params_dict_list)

import sys
import os
import datetime
import subprocess
import string
import numpy as np
import src.problemgenerator.series as series
import src.problemgenerator.array as array
import src.problemgenerator.filters as filter

# File format:
#     run_model_command ...
#     run_analyze_command ...
# special tokens:
#    [IN_i]: Will get replaced with input file i
#    [MID_j]: Will get replaced with unique filename, with suffix [MID_j]
#    [OUT_k]: Will get replaced with unique filename, with suffix [OUT_k]
def read_commands_file(commands_file_name):
    f = open(commands_file_name)
    run_model_command = f.readline().rstrip('\n')
    run_analyze_command = f.readline().rstrip('\n')
    return run_model_command, run_analyze_command

# def unique_filename(folder_name, file_name_suffix):
#     timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
#     current_folder = os.path.dirname(os.path.realpath(__file__))
#     return os.path.join(current_folder, "{}/{}-{}".format(folder_name, timestamp, file_name_suffix))

def unique_filename(folder, prefix, extension):
    current_folder = os.path.dirname(os.path.realpath(__file__))
    timestamp_string = str(datetime.datetime.utcnow().timestamp())
    fname = f"{current_folder}/{folder}/{prefix}_{timestamp_string}.{extension}"
    return fname

def substitute_tokens(string, token_signature, substitutions):
    res = string
    token_signature = "[" + token_signature + "_"
    appearances = string.count(token_signature)
    for i in range(appearances):
        tar = token_signature + str(i+1) + "]"
        res = res.replace(tar, substitutions[i])
    return res

def format_command(command, in_file_names, mid_file_names, out_file_names):
    res = command
    res = substitute_tokens(res, "IN", in_file_names)
    res = substitute_tokens(res, "MID", mid_file_names)
    res = substitute_tokens(res, "OUT", out_file_names)
    return res

def run_commands(run_model_command, run_analyze_command, in_file_names):
    n = run_model_command.count("[MID_") + run_analyze_command.count("[MID_")
    m = run_model_command.count("[OUT_") + run_analyze_command.count("[OUT_")
    mid_file_names = [unique_filename("tmp", "MID-" + str(i+1), "") for i in range(0, n)]
    out_file_names = [unique_filename("tmp", "OUT-" + str(i+1), "") for i in range(0, m)]
    
    command_1 = format_command(run_model_command, in_file_names, mid_file_names, out_file_names)
    command_2 = format_command(run_analyze_command, in_file_names, mid_file_names, out_file_names)
    print(command_1)
    print(command_2)

    subprocess.run(command_1, shell=True)
    subprocess.run(command_2, shell=True)

    return mid_file_names, out_file_names


def main():
    
    def save_errorified(std, prob):
        print(std, prob)
        x_node = array.Array(original_data[0][0].shape)
        x_node.addfilter(filter.GaussianNoise(0, std))
        x_node.addfilter(filter.Missing(prob))
        y_node = array.Array(original_data[1][0].shape)
        error_generator_root = series.TupleSeries([x_node, y_node])
        x_out, y_out = error_generator_root.process(original_data)
        x_name = unique_filename("tmp", "x", "npy")
        y_name = unique_filename("tmp", "y", "npy")
        np.save(x_name, x_out)
        np.save(y_name, y_out)
        return [x_name]

    # To be taken as arguments
    original_data_files = ["data/mnist_subset/x.npy", "data/mnist_subset/y.npy"]
    original_data = tuple([np.load(data_file) for data_file in original_data_files])
    commands_file_name = sys.argv[1]
    run_model_command, run_analyze_command = read_commands_file(commands_file_name)

    # To be read from file (file name given as argument)!
    n_output_datasets = 11
    std_vals = np.linspace(0.0, 1.0, n_output_datasets)
    prob_missing_vals = np.zeros((1,))

    for std in std_vals:
        for prob in prob_missing_vals:
            err_file_names = save_errorified(std, prob)
            mid_file_names, out_file_names = run_commands(run_model_command, run_analyze_command, err_file_names)

if __name__ == '__main__':
    main()

import sys
import os
import datetime
import subprocess
import string
import numpy as np
import src.problemgenerator.series as series
import src.problemgenerator.array as array
import src.problemgenerator.filters as filters
import re

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

def do_replacements(command, replacements):
    for key, value in replacements.items():
        command = command.replace(key, value)
    return command

def create_replacements(command, token_signature):
    regex = r'\[' + re.escape(token_signature) + r'_(\d*)\.(\w*)\]'
    matches = re.findall(regex, command)
    replacements = {}
    for pr in matches:
        replacements["[" + token_signature + "_" + pr[0] + "." + pr[1] + "]"] = unique_filename("tmp", token_signature + "-" + str(pr[0]), pr[1])
    return replacements

def run_commands(run_model_command, run_analyze_command, in_file_names):
    in_replacements = {'[IN_' + str(i+1) + ']': in_file_names[i] for i in range(0, len(in_file_names))}
    mid_replacements = create_replacements(run_model_command + run_analyze_command, "MID")
    out_replacements = create_replacements(run_model_command + run_analyze_command, "OUT")

    run_model_command = do_replacements(run_model_command, in_replacements)
    run_model_command = do_replacements(run_model_command, mid_replacements)
    run_model_command = do_replacements(run_model_command, out_replacements)

    run_analyze_command = do_replacements(run_analyze_command, in_replacements)
    run_analyze_command = do_replacements(run_analyze_command, mid_replacements)
    run_analyze_command = do_replacements(run_analyze_command, out_replacements)

    # print(run_model_command)
    # print(run_analyze_command)

    subprocess.run(command_1, shell=True)
    subprocess.run(command_2, shell=True)

    mid_file_names = [value for key, value in mid_replacements.items()]
    out_file_names = [value for key, value in out_replacements.items()]
    return mid_file_names, out_file_names


def main(): 
    def save_errorified(std, prob):
        print(std, prob)
        x_node = array.Array(original_data[0][0].shape)
        x_node.addfilter(filters.GaussianNoise(0, std))
        x_node.addfilter(filters.Missing(prob))
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

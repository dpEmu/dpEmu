import sys
import os
from datetime import datetime
import subprocess
import string
import numpy as np
import problemgenerator.series as series
import problemgenerator.array as array
import problemgenerator.filter as filter

# File format:
# 	run_model_command ...
# 	run_analyze_command ...
# special tokens:
#	[IN_i]: Will get replaced with input file i
#	[MID_j]: Will get replaced with unique filename, with suffix [MID_j]
#	[OUT_k]: Will get replaced with unique filename, with suffix [OUT_k]
def read_commands_file(commands_file_name):
	f = open(commands_file_name)
	run_model_command = f.readline().rstrip('\n')
	run_analyze_command = f.readline().rstrip('\n')
	return run_model_command, run_analyze_command

def unique_filename(folder_name, file_name_suffix):
	current_folder = os.path.dirname(os.path.realpath(__file__))
	timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
	return os.path.join(current_folder, "{}/{}-{}".format(folder_name, timestamp, file_name_suffix))

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
	mid_file_names = [unique_filename(".", "MID-" + str(i+1)) for i in range(0, n)]
	out_file_names = [unique_filename(".", "OUT-" + str(i+1)) for i in range(0, m)]
	
	command_1 = format_command(run_model_command, in_file_names, mid_file_names, out_file_names)
	command_2 = format_command(run_analyze_command, in_file_names, mid_file_names, out_file_names)
	print(command_1)
	print(command_2)

	subprocess.run(command_1, shell=True)
	subprocess.run(command_2, shell=True)

	return mid_file_names, out_file_names

# Placeholder
def create_err_files(i):
	return ["err_file_" + str(i)]
err_params_cou = 3

commands_file_name = sys.argv[1]
run_model_command, run_analyze_command = read_commands_file(commands_file_name)

for i in range(0, err_params_cou):
	err_file_names = create_err_files(i)
	int_file_names, out_file_names = run_commands(run_model_command, run_analyze_command, err_file_names)

def create_datestamped_filename(prefix, extension):
    timestamp_string = str(datetime.datetime.utcnow().timestamp())
    return f"{prefix}_{timestamp_string}.{extension}"

# To be read from file (file name given as argument)!
n_output_datasets = 11
std_vals = np.linspace(0.0, 1.0, n_output_datasets)
prob_missing_vals = np.zeros((1,))

# To be taken as arguments
original_data_files = ["../data/mnist_subset/x.npy", "../data/mnist_subset/y.npy"]

original_data = tuple([np.load(data_file) for data_file in original_data_files])

for std in std_vals:
    for prob in prob_missing_vals:
        print(std, prob)
        x_node = array.Array(original_data[0][0].shape)
        x_node.addfilter(filter.GaussianNoise(0, std))
        x_node.addfilter(filter.Missing(prob))
        y_node = array.Array(original_data[1][0].shape)
        error_generator_root = series.TupleSeries([x_node, y_node])
        x_out, y_out = error_generator_root.process(original_data)
        np.save(create_datestamped_filename("x", "npy"), x_out)
        np.save(create_datestamped_filename("y", "npy"), y_out)

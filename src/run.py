import sys
import numpy as np
import problemgenerator.series as series
import problemgenerator.array as array
import problemgenerator.filter as filter

# To be read from file (file name given as argument)!
std, prob_missing = .1, .2

# To be taken as arguments
original_data_files = ["../data/mnist_data.npy", "../data/mnist_label.npy"]

original_data = tuple([np.load(data_file) for data_file in original_data_files])

x_node = array.Array(original_data[0][0].shape)
x_node.addfilter(filter.GaussianNoise(0, std))
x_node.addfilter(filter.Missing(prob_missing))
y_node = array.Array(original_data[1][0].shape)
error_generator_root = series.TupleSeries([x_node, y_node])

errorified_data = error_generator_root.process(original_data)
#print(errorified_data[0].shape, errorified_data[1].shape)

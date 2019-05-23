import sys
import numpy as np
import src.problemgenerator.series as series

original_data_files = sys.argv[2:]

original_data = [np.load(data_file) for data_file in original_data_files]

error_generator_root = series.TupleSeries(original_data)
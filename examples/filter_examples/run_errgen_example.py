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

import numpy as np
from dpemu.nodes import Array, Series
from dpemu.filters.common import Missing

# To load data from a csv file, uncomment the rows below and
# give the data file name as the first command line argument.

# datafile = sys.argv[1]
# data = np.genfromtxt(datafile, delimiter=',')

# Suppose we have 10 sensors and 100 data points from each
# (each data point corresponding to, say, a different day)
observations, sensors = 100, 10

# Create a matrix of (random) data to use as input
data = np.random.randn(observations, sensors)

# Create an Array object to represent the battery of 10 sensors
sensor_array = Array()

# Add a Missing filters to randomly transform elements to Nan
# (NaN = "not a number", i.e. missing or invalid data)
sensor_array.addfilter(Missing("prob", "val"))

# Create a series to represent the 100 data points
root_node = Series(sensor_array)

# The data model tree is now complete.
# Process the data to introduce errors
output = root_node.generate_error(data, {'prob': .3, 'val': np.nan})

# Sanity check: does the shape of the output equal that of the input?
print("input data has shape", data.shape)
print("output data has shape", output.shape)

# The relative frequency on NaNs should be close to the probability
# given as a parameter to the Missing filters
print("relative frequency of NaNs:", np.isnan(output).sum() / output.size)

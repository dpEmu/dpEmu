import sys
import numpy as np
import problemgenerator.tensor as tensor
import problemgenerator.filter as filter
import problemgenerator.series as series

# To load data from a csv file, uncomment the rows below and
# give the data file name as the first command line argument.

# datafile = sys.argv[1]
# data = np.genfromtxt(datafile, delimiter=',')

# Suppose we have 10 sensors and 100 data points from each
# (each data point corresponding to, say, a different day)
observations, sensors = 100, 10

# Create fake (random) data
data = np.random.randn(observations, sensors)

# Create a Tensor to represent the battery of 10 sensors
t = tensor.Tensor(sensors)

# Add a Missing filter to randomly transform elements to Nan
# (NaN = "not a number", i.e. missing or invalid data)
t.addfilter(filter.Missing(probability=.3))

# Create a series to represent the 100 data points
s = series.Series(t)

# Process the data
out = s.process(data)

# Sanity check: does the shape of the output equal that of the input?
print("input data has shape", data.shape)
print("output data has shape", out.shape)

# The relative frequency on NaNs should be close to the probability
# given as a parameter to the Missing filter
print("relative frequency of NaNs:", np.isnan(out).sum() / out.size)
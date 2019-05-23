import numpy as np
import problemgenerator.array as array
import problemgenerator.filters as filters
import problemgenerator.series as series

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
sensor_array = array.Array(sensors)

# Add a Missing filters to randomly transform elements to Nan
# (NaN = "not a number", i.e. missing or invalid data)
sensor_array.addfilter(filters.Missing(probability=.3))

# Create a series to represent the 100 data points
observation_series = series.Series(sensor_array)

# The data model tree is now complete.
# Process the data to introduce errors
output = observation_series.process(data)

# Sanity check: does the shape of the output equal that of the input?
print("input data has shape", data.shape)
print("output data has shape", output.shape)

# The relative frequency on NaNs should be close to the probability
# given as a parameter to the Missing filters
print("relative frequency of NaNs:", np.isnan(output).sum() / output.size)

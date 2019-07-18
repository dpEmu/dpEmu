import numpy as np
import src.problemgenerator.array as array
import src.problemgenerator.filters as filters
import src.problemgenerator.series as series

# Assume our data is a tuple of the form (x, y) where x has
# shape (100, 10) and y has shape (100,). We can think of each
# row i as a data point where x_i represents the values of the
# explanatory variables and y_i represents the corresponding
# value of the response variable.
x = np.random.rand(100, 10)
y = np.random.rand(100, 1)
data = (x, y)

# Build a data model tree.
x_node = array.Array(x[0].shape)
y_node = array.Array(y[0].shape)
root_node = series.TupleSeries([x_node, y_node])

# Suppose we want to introduce NaN values (i.e. missing data)
# to y only (thus keeping x intact).
probability = .3
y_node.addfilter(filters.Missing("p"))

# Feed the data to the root node.
output = root_node.generate_error(data, {"p": probability})

print("Output type (should be tuple):", type(output))
print("Output length (should be 2):", len(output))
print("Shape of first member of output tuple (should be (100, 10)):",
      output[0].shape)
print("Shape of second first member of output tuple (should be (100,)):",
      output[1].shape)
print("Number of NaNs in x (should be 0):",
      np.isnan(output[0]).sum())
print(f"Number of NaNs in y (should be close to {probability * y.size}):",
      np.isnan(output[1]).sum())

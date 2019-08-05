import sys
import numpy as np
import matplotlib.pyplot as plt
import src.problemgenerator.array as array
import src.problemgenerator.filters as filters
import src.problemgenerator.series as series

"""
Generate time dependent Gaussian noise and (non time dependent) missing values to MNIST data.
Command line arguments: standard_deviation, standard_deviation_increase, missing_probability
"""

std = float(sys.argv[1])
std_increase = float(sys.argv[2])
prob = float(sys.argv[3])

x_file, y_file = "data/mnist_subset/x.npy", "data/mnist_subset/y.npy"
x = np.load(x_file)
y = np.load(y_file)

params = {}
params["mean"] = 0
params["std"] = std
params["mean_inc"] = 0
params["std_inc"] = std_increase
params["p"] = prob


x_node = array.Array()
x_node.addfilter(filters.GaussianNoiseTimeDependent("mean", "std", "mean_inc", "std_inc"))
x_node.addfilter(filters.Missing("p"))
y_node = array.Array(y[0].shape)
root_node = series.TupleSeries([x_node, y_node], dim_name="time")
out_x, out_y = root_node.generate_error((x, y), params)

print((y != out_y).sum(), "elements of y have been modified in (should be 0).")

examples = 4
fig, axs = plt.subplots(2, examples)
for i in range(examples):
    img_ind = np.random.randint(len(x))
    axs[0, i].matshow(x[img_ind], cmap='gray_r')
    axs[0, i].axis('off')
    axs[1, i].matshow(out_x[img_ind], cmap='gray_r')
    axs[1, i].axis('off')

plt.show()
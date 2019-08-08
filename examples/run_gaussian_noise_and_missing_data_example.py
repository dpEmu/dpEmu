import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from dpemu import array
from dpemu import filters
from dpemu import series

std = float(sys.argv[1])
prob = float(sys.argv[2])

n = 1000
x_file = Path("data/mnist_data.npy")
y_file = Path("data/mnist_label.npy")
x = np.load(x_file)[:n].reshape((n, 28, 28))
y = np.load(y_file)[:n]

x_node = array.Array()
x_node.addfilter(filters.GaussianNoise("mean", "std"))
x_node.addfilter(filters.Missing("prob"))
y_node = array.Array()
root_node = series.TupleSeries([x_node, y_node])
error_params = {"mean": 0, "std": std, "prob": prob}
out_x, out_y = root_node.generate_error((x, y), error_params)

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

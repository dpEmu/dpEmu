import sys
import numpy as np
import matplotlib.pyplot as plt
import problemgenerator.array as array
import problemgenerator.filters as filters
import problemgenerator.series as series

std = float(sys.argv[1])
prob_missing = float(sys.argv[2])

x_file, y_file = "data/mnist_subset/x.npy", "data/mnist_subset/y.npy"
x = np.load(x_file)
y = np.load(y_file)
data = (x, y)

x_node = array.Array(x[0].shape)
x_node.addfilter(filters.GaussianNoise(0, std))
x_node.addfilter(filters.Missing(prob_missing))
y_node = array.Array(y[0].shape)
root_node = series.TupleSeries([x_node, y_node])

out_x, out_y = root_node.process(data)
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

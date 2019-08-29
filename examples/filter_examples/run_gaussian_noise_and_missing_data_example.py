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

import sys
import numpy as np
import matplotlib.pyplot as plt
from dpemu.nodes import Array, TupleSeries
from dpemu.filters.common import GaussianNoise, Missing
from dpemu.dataset_utils import load_mnist

std = float(sys.argv[1])
prob = float(sys.argv[2])

n = 1000
x, y, _, _ = load_mnist(reshape_to_28x28=True)
x = x[:n].astype('float')
y = y[:n]
print(x.min(), x.max())

x_node = Array()
x_node.addfilter(GaussianNoise("mean", "std"))
x_node.addfilter(Missing("prob", "missing_value"))
y_node = Array()
root_node = TupleSeries([x_node, y_node])
error_params = {"mean": 0, "std": std, "prob": prob, "missing_value": np.nan}
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

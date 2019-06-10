import sys
import numpy as np
import matplotlib.pyplot as plt
from src.problemgenerator.add_noise import add_noise_to_imgs

std = float(sys.argv[1])
out = add_noise_to_imgs("data/mnist_subset/x.npy", "data/mnist_subset/y.npy", std, np.random.RandomState(42))
out_x, out_y = out[0][0], out[0][1]
x, y = out[1][0], out[1][1]
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

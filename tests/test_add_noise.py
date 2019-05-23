import numpy as np
from src.problemgenerator.add_noise import add_noise_to_imgs

def test_add_noise_to_imgs_with_zero_std():
    x_file, y_file = "data/mnist_subset/x.npy", "data/mnist_subset/y.npy"
    std = 0
    x = np.load(x_file)
    y = np.load(y_file)
    results = add_noise_to_imgs(x_file, y_file, std)

    assert np.array_equal(x, results[0][0])
    assert np.array_equal(y, results[0][1])
    assert np.array_equal(x, results[1][0])
    assert np.array_equal(y, results[1][1])

def test_add_noise_to_imgs():
    x_file, y_file = "data/mnist_subset/x.npy", "data/mnist_subset/y.npy"
    std = 0.2
    x = np.load(x_file)
    y = np.load(y_file)
    results = add_noise_to_imgs(x_file, y_file, std)

    assert not np.array_equal(x, results[0][0])
    assert np.array_equal(y, results[0][1])
    assert np.array_equal(x, results[1][0])
    assert np.array_equal(y, results[1][1])

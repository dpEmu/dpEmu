import os.path
import numpy as np

from sklearn.datasets import fetch_openml

data_filename = "mnist_data.npy"
label_filename = "mnist_label.npy"

def fetch_mnist():
    print("Fetching MNIST data")
    mnist = fetch_openml('mnist_784')
    data = mnist['data']
    target = mnist['target'].astype(int)

    print("Saving data to " + data_filename + " and " + label_filename)

    np.save(data_filename, data)
    np.save(label_filename, target)

if __name__ == "__main__":
    if os.path.isfile(data_filename) and os.path.isfile(label_filename):
        print("MNIST data already exists in this folder")
    else:
        fetch_mnist()

import os.path

from sklearn.datasets import fetch_openml

data_filename = "mnist_data.txt"
label_filename = "mnist_label.txt"

def fetch_mnist():
    print("Fetching MNIST data")
    mnist = fetch_openml('mnist_784')
    data = mnist['data']
    target = mnist['target'].astype(int)

    print("Saving data to " + data_filename + " and " + label_filename)

    data.tofile(data_filename)
    target.tofile(label_filename)

if __name__ == "__main__":
    if os.path.isfile(data_filename) and os.path.isfile(label_filename):
        print("MNIST data already exists in this folder")
    else:
        fetch_mnist()

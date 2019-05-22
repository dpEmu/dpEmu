from sklearn.datasets import fetch_openml
import json
import os.path

data_filename = "mnist_data.json"
label_filename = "mnist_label.json"

def fetch_mnist():
    print("Fetching MNIST data")
    mnist = fetch_openml('mnist_784')
    data = mnist['data']
    target = mnist['target'].astype(int)
    
    data_obj = {"data": data.tolist()}
    label_obj = {"labels": target.tolist()}

    print("Saving data to " + data_filename + " and " + label_filename)

    data_file = open(data_filename, 'w')
    label_file = open(label_filename, 'w')

    data_file.write(json.dumps(data_obj))
    label_file.write(json.dumps(label_obj))

    data_file.close()
    label_file.close()

if __name__ == "__main__":
    if os.path.isfile(data_filename) and os.path.isfile(label_filename):
        print("MNIST data already exists in this folder")
    else:
        fetch_mnist()

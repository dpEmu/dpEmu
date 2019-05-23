import json
import os
import subprocess
from datetime import datetime

import numpy as np
from PIL import Image
from sklearn.datasets import load_digits, fetch_openml
from sklearn.model_selection import train_test_split


def load_digits_to_npy():
    digits = load_digits()
    path_to_data = generate_unique_path_prefix("data", "digits_data") + ".npy"
    path_to_labels = generate_unique_path_prefix("data", "digits_labels") + ".npy"
    save_dataset(path_to_data, path_to_labels, digits["data"], digits["target"])
    return path_to_data, path_to_labels


def load_mnist_to_npy():
    mnist = fetch_openml("mnist_784")
    data, _, labels, _ = train_test_split(mnist["data"], mnist["target"].astype(int), train_size=.3, random_state=42)
    path_to_data = generate_unique_path_prefix("data", "mnist_data") + ".npy"
    path_to_labels = generate_unique_path_prefix("data", "mnist_labels") + ".npy"
    save_dataset(path_to_data, path_to_labels, data, labels)
    return path_to_data, path_to_labels


def save_dataset(path_to_data, path_to_labels, data, labels):
    if not os.path.isfile(path_to_data):
        np.save(path_to_data, data)
        np.save(path_to_labels, labels)


def generate_unique_path_prefix(folder_name, file_name=None):
    current_folder = os.path.dirname(os.path.realpath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    if file_name:
        unique_path_prefix = os.path.join(current_folder, "{}/{}".format(folder_name, file_name))
    else:
        unique_path_prefix = os.path.join(current_folder, "{}/{}".format(folder_name, timestamp))
    return unique_path_prefix


def run_script(args):
    prog = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    err = prog.communicate()[1]
    if err:
        print(err)


def main():
    path_to_data, path_to_labels = load_digits_to_npy()
    path_to_reduced_data = generate_unique_path_prefix("tmp") + ".npy"
    path_to_fitted_model = generate_unique_path_prefix("tmp") + ".joblib"
    path_to_result_prefix = generate_unique_path_prefix("out")
    path_to_classes_img = path_to_result_prefix + "_classes.png"
    path_to_clusters_img = path_to_result_prefix + "_clusters.png"
    path_to_scores = path_to_result_prefix + "_scores.json"

    kmeans_model_params = {
        "path_to_data": path_to_data,
        "path_to_labels": path_to_labels,
        "path_to_reduced_data": path_to_reduced_data,
        "path_to_fitted_model": path_to_fitted_model,
    }
    path_to_kmeans_model_params = generate_unique_path_prefix("tmp") + ".json"
    with open(path_to_kmeans_model_params, "w") as fp:
        json.dump(kmeans_model_params, fp)

    run_script("python src/ml/kmeans_model.py {}".format(path_to_kmeans_model_params).split())

    clustering_analyzer_params = {
        "path_to_reduced_data": path_to_reduced_data,
        "path_to_labels": path_to_labels,
        "path_to_fitted_model": path_to_fitted_model,
        "path_to_classes_img": path_to_classes_img,
        "path_to_clusters_img": path_to_clusters_img,
        "path_to_scores": path_to_scores,
    }
    path_to_clustering_analyzer_params = generate_unique_path_prefix("tmp") + ".json"
    with open(path_to_clustering_analyzer_params, "w") as fp:
        json.dump(clustering_analyzer_params, fp)

    run_script("python src/ml/clustering_analyzer.py {}".format(path_to_clustering_analyzer_params).split())

    with open(path_to_scores, "r") as fp:
        scores = json.load(fp)
    classes_img = Image.open(path_to_classes_img)
    clusters_img = Image.open(path_to_clusters_img)

    print(scores)
    classes_img.show()
    clusters_img.show()


if __name__ == "__main__":
    main()

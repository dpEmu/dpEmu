import json

from PIL import Image

from src.ml.utils import run_ml_script
from src.utils import load_digits_as_npy, load_mnist_as_npy, generate_unique_path


def main():
    path_to_data, path_to_labels = load_digits_as_npy()
    # path_to_data, path_to_labels = load_mnist_as_npy(70000)
    path_to_reduced_data = generate_unique_path("tmp", "npy")
    path_to_fitted_model = generate_unique_path("tmp", "joblib")
    path_to_classes_img = generate_unique_path("tmp", "png")
    path_to_clusters_img = generate_unique_path("tmp", "png")
    path_to_scores = generate_unique_path("tmp", "json")

    run_ml_script("python src/ml/kmeans_model.py {} {} {} {}".format(path_to_data, path_to_labels, path_to_reduced_data,
                                                                     path_to_fitted_model))
    run_ml_script("python src/ml/clustering_analyzer.py {} {} {} {} {} {}".format(path_to_reduced_data, path_to_labels,
                                                                                  path_to_fitted_model,
                                                                                  path_to_classes_img,
                                                                                  path_to_clusters_img, path_to_scores))

    with open(path_to_scores, "r") as file:
        scores = json.load(file)
    classes_img = Image.open(path_to_classes_img)
    clusters_img = Image.open(path_to_clusters_img)

    print(scores)
    classes_img.show()
    clusters_img.show()


if __name__ == "__main__":
    main()

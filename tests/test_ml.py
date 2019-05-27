import json

from PIL import Image

from src.ml.utils import run_clustering_model_script, run_clustering_analyzer_script
from src.utils import load_digits_as_npy, generate_unique_path


def test_kmeans_with_analysis():
    path_to_data, path_to_labels = load_digits_as_npy()
    path_to_reduced_data = generate_unique_path("tmp", "npy")
    path_to_fitted_model = generate_unique_path("tmp", "joblib")
    path_to_classes_img = generate_unique_path("tmp", "png")
    path_to_clusters_img = generate_unique_path("tmp", "png")
    path_to_scores = generate_unique_path("tmp", "json")

    clustering_params = {
        "path_to_data": path_to_data,
        "path_to_labels": path_to_labels,
        "path_to_reduced_data": path_to_reduced_data,
        "path_to_fitted_model": path_to_fitted_model,
        "path_to_classes_img": path_to_classes_img,
        "path_to_clusters_img": path_to_clusters_img,
        "path_to_scores": path_to_scores,
    }
    run_clustering_model_script("kmeans", clustering_params)
    run_clustering_analyzer_script(clustering_params)

    with open(path_to_scores, "r") as file:
        scores = json.load(file)
    Image.open(path_to_classes_img).verify()
    Image.open(path_to_clusters_img).verify()

    assert scores == {"v-meas": "0.905", "ARI": "0.822", "AMI": "0.904", "silhouette": "0.792"}

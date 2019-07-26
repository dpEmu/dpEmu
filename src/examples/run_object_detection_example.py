import os
import re
import subprocess
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
from PIL import Image
from numpy.random import RandomState

from src import runner_
from src.datasets.utils import load_coco_val_2017
from src.plotting.utils import print_results, visualize_scores
from src.problemgenerator.array import Array
from src.problemgenerator.filters import JPEG_Compression
from src.problemgenerator.series import Series


class Preprocessor:
    def run(self, _, imgs):
        return None, imgs, {}


def write_imgs_to_disk(imgs, img_filenames):
    for i, img_arr in enumerate(imgs):
        img = Image.fromarray(img_arr)
        path_to_img = "tmp/val2017/" + img_filenames[i]
        img.save(path_to_img, "jpeg", quality=100)


def get_map_score():
    with open("tmp/results.txt", "r") as file:
        text = file.read()
    return float(re.findall(r"[-+]?\d*\.\d+", text)[1])


class YOLOv3GPUModel:

    def __init__(self):
        self.random_state = RandomState(42)

    def run(self, _, imgs, params):
        img_filenames = params["img_filenames"]

        path_to_yolov3_weights = "tmp/yolov3-spp_best.weights"
        if not os.path.isfile(path_to_yolov3_weights):
            subprocess.call(["./scripts/get_yolov3.sh"])

        write_imgs_to_disk(imgs, img_filenames)
        subprocess.call(["./scripts/run_darknet.sh"])

        return {"mAP-50": round(get_map_score(), 3)}


class AbstractDetectronModel(ABC):

    def __init__(self):
        self.random_state = RandomState(42)

    def run(self, _, imgs, params):
        img_filenames = params["img_filenames"]
        path_to_cfg = self.get_path_to_cfg()
        url_to_weights = self.get_url_to_weights()

        write_imgs_to_disk(imgs, img_filenames)
        subprocess.call(["./scripts/run_detectron.sh", path_to_cfg, url_to_weights])

        return {"mAP-50": round(get_map_score(), 3)}

    @abstractmethod
    def get_path_to_cfg(self):
        pass

    @abstractmethod
    def get_url_to_weights(self):
        pass


class FasterRCNNModel(AbstractDetectronModel):
    def __init__(self):
        super().__init__()

    def get_path_to_cfg(self):
        return "libs/Detectron/configs/12_2017_baselines/e2e_faster_rcnn_X-101-64x4d-FPN_1x.yaml"

    def get_url_to_weights(self):
        return (
            "https://dl.fbaipublicfiles.com/detectron/35858015/12_2017_baselines/"
            "e2e_faster_rcnn_X-101-64x4d-FPN_1x.yaml.01_40_54.1xc565DE/output/train/"
            "coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl"
        )


class MaskRCNNModel(AbstractDetectronModel):
    def __init__(self):
        super().__init__()

    def get_path_to_cfg(self):
        return "libs/Detectron/configs/12_2017_baselines/e2e_mask_rcnn_X-101-64x4d-FPN_1x.yaml"

    def get_url_to_weights(self):
        return (
            "https://dl.fbaipublicfiles.com/detectron/36494496/12_2017_baselines/"
            "e2e_mask_rcnn_X-101-64x4d-FPN_1x.yaml.07_50_11.fkwVtEvg/output/train/"
            "coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl"
        )


class RetinaNetModel(AbstractDetectronModel):
    def __init__(self):
        super().__init__()

    def get_path_to_cfg(self):
        return "libs/Detectron/configs/12_2017_baselines/retinanet_X-101-64x4d-FPN_1x.yaml"

    def get_url_to_weights(self):
        return (
            "https://dl.fbaipublicfiles.com/detectron/36768875/12_2017_baselines/"
            "retinanet_X-101-64x4d-FPN_1x.yaml.08_34_37.FSXgMpzP/output/train/"
            "coco_2014_train%3Acoco_2014_valminusminival/retinanet/model_final.pkl"
        )


def visualize(df):
    # visualize_scores(df, ["mAP-50"], [True], "std", "Object detection with Gaussian noise", log=False)
    # visualize_scores(df, ["mAP-50"], [True], "std", "Object detection with Gaussian blur", log=False)
    # visualize_scores(df, ["mAP-50"], [True], "snowflake_probability", "Object detection with snow filter", log=True)
    # visualize_scores(df, ["mAP-50"], [True], "probability", "Object detection with rain filter", log=True)
    # visualize_scores(df, ["mAP-50"], [True], "probability", "Object detection with added stains", log=True)
    visualize_scores(df, ["mAP-50"], [True], "quality", "Object detection with JPEG compression", log=False)

    plt.show()


def main():
    imgs, _, _, img_filenames = load_coco_val_2017()

    err_node = Array()
    err_root_node = Series(err_node)

    # err_node.addfilter(GaussianNoise("mean", "std"))
    # err_node.addfilter(Blur_Gaussian("std"))
    # err_node.addfilter(Snow("snowflake_probability", "snowflake_alpha", "snowstorm_alpha"))
    # err_node.addfilter(Rain("probability"))
    # err_node.addfilter(StainArea("probability", "radius_generator", "transparency_percentage"))
    err_node.addfilter(JPEG_Compression("quality"))
    # err_node.addfilter(Identity())

    # err_params_list = [{"mean": 0, "std": std} for std in [10 * i for i in range(0, 4)]]
    # err_params_list = [{"std": std} for std in [i for i in range(0, 4)]]
    # err_params_list = [{"snowflake_probability": p, "snowflake_alpha": .4, "snowstorm_alpha": 0}
    #                    for p in [10 ** i for i in range(-4, 0)]]
    # err_params_list = [{"probability": p} for p in [10 ** i for i in range(-4, 0)]]
    # err_params_list = [
    #     {"probability": p, "radius_generator": GaussianRadiusGenerator(0, 50), "transparency_percentage": 0.2}
    #     for p in [10 ** i for i in range(-6, -2)]]
    err_params_list = [{"quality": q} for q in [10, 20, 30, 100]]
    # err_params_list = [{}]

    model_params_dict_list = [
        {"model": FasterRCNNModel, "params_list": [{"img_filenames": img_filenames}]},
        {"model": MaskRCNNModel, "params_list": [{"img_filenames": img_filenames}]},
        {"model": RetinaNetModel, "params_list": [{"img_filenames": img_filenames}]},
        {"model": YOLOv3GPUModel, "params_list": [{"img_filenames": img_filenames}]},
    ]

    df = runner_.run(None, imgs, Preprocessor, err_root_node, err_params_list, model_params_dict_list, n_processes=1)

    print_results(df, ["img_ids", "img_filenames", "show_imgs", "mean", "std", "radius_generator",
                       "transparency_percentage"])
    visualize(df)


if __name__ == "__main__":
    main()

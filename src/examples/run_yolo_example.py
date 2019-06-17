import json
import os
import subprocess
from copy import deepcopy

import cv2
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from src import runner
from src.problemgenerator import array, copy, filters
from src.utils import generate_unique_path


class Model:

    def __init__(self):
        cv2.setNumThreads(1)
        self.results = []
        self.coco91class = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33,
            34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
            62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90
        ]
        with open("data/coco.names", "r") as fp:
            self.class_names = [line.strip() for line in fp.readlines()]

    def __print_result(self, result):
        result = deepcopy(result)
        result["category_id"] = self.class_names[self.coco91class.index(result["category_id"])]
        print(result)

    def __add_img_to_results(self, img, img_id):
        conf_threshold = .25
        img_h = img.shape[0]
        img_w = img.shape[1]
        inference_size = 416
        scale = 1 / 255

        net = cv2.dnn.readNet("data/yolov3.weights", "data/yolov3.cfg")
        blob = cv2.dnn.blobFromImage(img, scale, (inference_size, inference_size), (0, 0, 0), True, crop=False)
        net.setInput(blob)

        out_layer_names = net.getUnconnectedOutLayersNames()
        outs = net.forward(out_layer_names)

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                conf = round(scores[class_id], 3)
                if conf > conf_threshold:
                    center_x = detection[0] * img_w
                    center_y = detection[1] * img_h
                    w = round(detection[2] * img_w, 2)
                    h = round(detection[3] * img_h, 2)
                    x = round(center_x - w / 2, 2)
                    y = round(center_y - h / 2, 2)

                    result = {
                        "image_id": img_id,
                        "category_id": self.coco91class[class_id],
                        "bbox": [x, y, w, h],
                        "score": float(conf),
                    }
                    self.results.append(result)
                    # self.__print_result(result)

    def run(self, imgs, model_params):
        img_ids = model_params["img_ids"]

        [self.__add_img_to_results(imgs[i], img_ids[i]) for i in tqdm(range(len(imgs)))]

        path_to_results = generate_unique_path("tmp", "json")
        with open(path_to_results, "w") as fp:
            json.dump(self.results, fp)

        coco_gt = COCO("data/annotations/instances_val2017.json")
        coco_eval = COCOeval(coco_gt, coco_gt.loadRes(path_to_results), "bbox")
        coco_eval.params.imgIds = img_ids
        # coco_eval.params.imgIds = [139]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return {"mAP-50": round(coco_eval.stats[1], 3)}


def load_coco_val_2017():
    img_folder = "data/val2017"
    if not os.path.isdir(img_folder):
        subprocess.call(["./data/get_coco_dataset.sh"])
    path_to_yolov3_weights = "data/yolov3.weights"
    if not os.path.isfile(path_to_yolov3_weights):
        subprocess.call(["./data/get_yolov3.sh"])

    coco = COCO("data/annotations/instances_val2017.json")
    # img_ids = sorted(coco.getImgIds())
    img_ids = sorted(coco.getImgIds())[:2]
    img_dicts = coco.loadImgs(img_ids)
    # img_dicts = coco.loadImgs([139])
    imgs = [cv2.imread(os.path.join(img_folder, img_dict["file_name"])) for img_dict in img_dicts]
    return imgs, img_ids


class ErrGen:
    def __init__(self, imgs):
        self.imgs = imgs

    def generate_error(self, params):
        imgs = deepcopy(self.imgs)
        results = []
        for img in imgs:
            img_node = array.Array(img.shape)
            root_node = copy.Copy(img_node)
            img_node.addfilter(filters.GaussianNoise(params["mean"], params["std"]))
            result = root_node.process(img.astype(float), np.random.RandomState(seed=42))
            results.append(result)

            cv2.imshow("GaussianNoise: mean {}, std {}".format(params["mean"], params["std"]), np.uint8(result))
            cv2.waitKey()
            cv2.destroyAllWindows()
        return results


class ParamSelector:
    def __init__(self, params):
        self.params = params

    def next(self):
        return self.params

    def analyze(self, res):
        self.params = None


def main():
    imgs, img_ids = load_coco_val_2017()
    model = Model()

    err_gen = ErrGen(imgs)
    param_selector = ParamSelector(
        [({"mean": a, "std": b}, {"img_ids": img_ids}) for (a, b) in [(0, 0), (0, 15), (0, 30)]])
    out = runner.run(model, err_gen, param_selector)

    # out = model.run(imgs, {"img_ids": img_ids})

    print(out)


if __name__ == "__main__":
    main()

import json
import os
import subprocess

import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.utils import generate_unique_path


class Model:

    def __init__(self):
        self.results = []
        self.coco91class = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33,
            34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
            62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90
        ]

    def __add_img_to_results(self, img, img_id):
        conf_threshold = .5
        img_h = img.shape[0]
        img_w = img.shape[1]
        inf_size = 416
        scale = 0.00392

        net = cv2.dnn.readNet("data/yolov3.weights", "data/yolov3.cfg")
        blob = cv2.dnn.blobFromImage(img, scale, (inf_size, inf_size), (0, 0, 0), True, crop=False)
        net.setInput(blob)

        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        outs = net.forward(output_layers)

        for out in outs:
            for obj in out:
                scores = obj[5:]
                class_id = np.argmax(scores)
                conf = round(float(scores[class_id]), 3)
                if conf > conf_threshold:
                    center_x = float(obj[0]) * img_w
                    center_y = float(obj[1]) * img_h
                    w = round(float(obj[2]) * img_w, 2)
                    h = round(float(obj[3]) * img_h, 2)
                    x = round(center_x - w / 2, 2)
                    y = round(center_y - h / 2, 2)

                    res = {
                        "image_id": img_id,
                        "category_id": self.coco91class[class_id],
                        "bbox": [x, y, w, h],
                        "score": conf,
                    }
                    print(res)
                    self.results.append(res)

    def run(self, data):
        [self.__add_img_to_results(img, img_id) for img, img_id in data]

        path_to_results = generate_unique_path("tmp", "json")
        with open(path_to_results, "w") as fp:
            json.dump(self.results, fp)

        coco_gt = COCO("data/annotations/instances_val2017.json")
        coco_eval = COCOeval(coco_gt, coco_gt.loadRes(path_to_results), "bbox")
        coco_eval.params.imgIds = [img_id for _, img_id in data]
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
    img_ids = sorted(coco.getImgIds())[:10]
    img_dicts = coco.loadImgs(img_ids)
    imgs = [cv2.imread(os.path.join(img_folder, img_dict["file_name"])) for img_dict in img_dicts]
    return [(imgs[i], img_ids[i]) for i in range(len(imgs))]


def main():
    data = load_coco_val_2017()
    model = Model()
    out = model.run(data)
    print(out)


if __name__ == "__main__":
    main()

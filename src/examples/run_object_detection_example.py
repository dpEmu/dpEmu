import cv2
import detectron.utils.c2 as c2_utils
from caffe2.python import workspace
from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.core.config import merge_cfg_from_list
from detectron.core.test_engine import run_inference
from detectron.utils.logging import setup_logging

c2_utils.import_detectron_ops()
cv2.ocl.setUseOpenCL(False)


def main():
    workspace.GlobalInit(["caffe2", "--caffe2_log_level=0"])
    setup_logging(__name__)

    merge_cfg_from_file("venv/src/detectron/configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml")
    url_to_weights = (
        "https://dl.fbaipublicfiles.com/detectron/35861858/12_2017_baselines/"
        "e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/output/train/coco_2014_train:coco_2014_valminusminival/"
        "generalized_rcnn/model_final.pkl"
    )
    opt_list = [
        "MODEL.KEYPOINTS_ON",
        False,
        "MODEL.MASK_ON",
        False,
        "NUM_GPUS",
        "1",
        "TEST.DATASETS",
        ("coco_2017_val",),
        "TEST.SCALE",
        "416",
        "TEST.WEIGHTS",
        url_to_weights,
        "OUTPUT_DIR",
        "tmp"
    ]
    merge_cfg_from_list(opt_list)
    assert_and_infer_cfg()

    results = run_inference(cfg.TEST.WEIGHTS)
    print(round(results["coco_2017_val"]["box"]["AP50"], 3))


if __name__ == "__main__":
    main()

import cv2
import detectron.utils.c2 as c2_utils
from caffe2.python import workspace
from detectron.core.config import cfg, merge_cfg_from_file, assert_and_infer_cfg
from detectron.core.test_engine import run_inference

c2_utils.import_detectron_ops()
cv2.ocl.setUseOpenCL(False)


def main():
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    merge_cfg_from_file("$DETECTRON/configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml")
    assert_and_infer_cfg()
    run_inference(cfg.TEST.WEIGHTS, check_expected_results=True)


if __name__ == "__main__":
    main()

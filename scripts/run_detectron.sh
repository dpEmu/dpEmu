#!/bin/bash

python libs/Detectron/tools/test_net.py \
    --cfg $1 \
    TEST.WEIGHTS $2 \
    NUM_GPUS 1 \
    TEST.DATASETS '("coco_2017_val",)' \
    MODEL.MASK_ON False \
    OUTPUT_DIR tmp \
    | grep "IoU=0.50 " > tmp/results.txt

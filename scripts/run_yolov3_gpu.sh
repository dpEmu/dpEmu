#!/bin/bash

./libs/darknet/darknet detector map data/coco.data tmp/yolov3-spp.cfg tmp/yolov3-spp_best.weights | grep mAP@0.50 > results.txt

#!/bin/bash

cd libs/darknet
./darknet detector map cfg/coco.data cfg/yolov3-spp.cfg ../../tmp/yolov3-spp_best.weights | grep mAP@0.50 > ../../tmp/results.txt

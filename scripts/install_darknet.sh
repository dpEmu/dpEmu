#!/bin/bash

cd libs/darknet
make
wget -c https://www.dropbox.com/s/jxr17c1bt6ih9oy/yolov3-spp_best.weights
mkdir coco/images/val2017

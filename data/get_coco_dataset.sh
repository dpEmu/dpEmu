#!/bin/bash

cd data

wget -c https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget -c http://images.cocodataset.org/zips/val2017.zip

unzip -q annotations_trainval2017.zip
unzip -q val2017.zip

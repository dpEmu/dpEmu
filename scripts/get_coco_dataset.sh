#!/bin/bash

cd data

wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget -c http://images.cocodataset.org/zips/val2017.zip

unzip annotations_trainval2017.zip
unzip val2017.zip

rm annotations_trainval2017.zip
rm val2017.zip

cd ..
mkdir tmp/val2017
mkdir data/labels
java -jar data/cocotoyolo.jar "data/annotations/instances_val2017.json" "$PWD/tmp/images/val2017/" "all" "tmp/labels/val2017/"

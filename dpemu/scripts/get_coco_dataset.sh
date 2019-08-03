#!/bin/bash

cd data

wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget -c http://images.cocodataset.org/zips/val2017.zip
wget -c https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names

unzip annotations_trainval2017.zip
unzip val2017.zip

rm annotations_trainval2017.zip
rm val2017.zip

mkdir val2017_labels

cd ..
wget -c http://commecica.com/wp-content/uploads/2018/07/cocotoyolo.jar
java -jar cocotoyolo.jar "data/annotations/instances_val2017.json" "$PWD/tmp/val2017/" "all" "data/val2017_labels/"
mv data/val2017_labels/image_list.txt data/val2017.txt

mkdir -p tmp/val2017

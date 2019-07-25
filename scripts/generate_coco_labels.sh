#!/bin/bash

mkdir -p tmp/images/val2017
mkdir -p tmp/labels/val2017
java -jar data/cocotoyolo.jar "data/annotations/instances_val2017.json" "$PWD/tmp/images/val2017/" "all" "tmp/labels/val2017/"

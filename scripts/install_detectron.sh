#!/bin/bash

cd libs/Detectron
python setup.py develop
ln -s $PWD/../.. $PWD/detectron/datasets/data/coco

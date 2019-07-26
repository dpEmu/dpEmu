#!/bin/bash

cd libs/Detectron
make dev
ln -s ../.. $PWD/detectron/datasets/data/coco

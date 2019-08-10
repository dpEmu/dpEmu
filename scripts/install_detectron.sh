#!/bin/bash

if [ -z "$TMPDIR" ]; then
  pip install -r libs/Detectron/requirements.txt
else
  pip install -r libs/Detectron/requirements.txt --cache-dir $TMPDIR
fi

pip install -e libs/Detectron

FILE=$PWD/libs/Detectron/detectron/datasets/data/coco
if [ ! -e "$FILE" ]; then
  ln -s $PWD $FILE
fi

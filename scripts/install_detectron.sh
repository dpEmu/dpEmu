#!/bin/bash

if [ -z "$TMPDIR" ]; then
  pip install -r libs/Detectron/requirements.txt
  pip install -e libs/Detectron
else
  pip install -r libs/Detectron/requirements.txt --cache-dir $TMPDIR
  pip install -e libs/Detectron --cache-dir $TMPDIR
fi

FILE=$PWD/libs/Detectron/detectron/datasets/data/coco
if [ ! -e "$FILE" ]; then
  ln -s $PWD $FILE
fi

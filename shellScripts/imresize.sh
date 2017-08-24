#!/bin/bash

for img in *.pgm; do
    echo "resizing $img to 1024"
    convert $img -resize 1024x1024 rs$img

done

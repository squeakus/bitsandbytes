#!/bin/bash
list=`find . -name "*.JPG"`

for img in $list; do
    echo "img: $img"
    exiftool $img | grep 'Relative Altitude'
done

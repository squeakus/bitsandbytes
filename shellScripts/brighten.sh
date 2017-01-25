#!/bin/bash
list=`ls | grep .jpg` 
for img in $list; do
    echo "brightening $img"
    convert $img -modulate 150% $img
done


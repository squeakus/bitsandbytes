#!/bin/bash
list=`ls | grep .gif` 
for img in $list; do
    echo "resizing $img"
    mogrify -resize 75% $img
done


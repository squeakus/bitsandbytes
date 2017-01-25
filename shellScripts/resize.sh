#!/bin/bash
list=`ls | grep '.jpg\|.JPG'` 
for img in $list; do
    echo "resizing $img"
    mogrify -resize 33% $img
done


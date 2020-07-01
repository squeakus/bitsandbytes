#!/bin/bash
list=`ls | grep '.jpg\|.JPG\|.png\|.PNG\|.tif\|.TIF'` 
for img in $list; do
    echo "rejigging $img"
    # mogrify -resize 33% $img
    # mogrify -resize 100x100 $img # the ! ignores aspect ratio
    python3 color_extractor.py -i $img
done


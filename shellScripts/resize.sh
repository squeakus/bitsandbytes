#!/bin/bash
echo "HELLO!"

list=`ls | grep '.jpg\|.JPG\|.png\|.PNG\|.tif\|.TIF'` 
for img in $list; do
    echo "resizing $img"
    # mogrify -resize 33% $img
    # mogrify -resize 100x100 $img # the ! ignores aspect ratio
    mogrify -resize 1000x1000! $img
done


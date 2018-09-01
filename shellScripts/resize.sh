#!/bin/bash
list=`ls | grep '.jpg\|.JPG'` 
for img in $list; do
    echo "resizing $img"
    # mogrify -resize 33% $img
    mogrify -resize 640x480! $img # the ! ignores aspect ratio
done


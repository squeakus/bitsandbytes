#!/bin/bash
list=`ls | grep '.jpg\|.JPG\|.png\|.PNG'` 
for img in $list; do
    echo "resizing $img"
    # mogrify -resize 33% $img
    mogrify -resize 512x512! $img # the ! ignores aspect ratio
done


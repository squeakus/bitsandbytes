#!/bin/bash
list=`ls | grep .JPG` 
for img in $list; do
    echo "tidying up $img"
    mogrify -rotate "+2>" $img
    mogrify -crop "-100-105" $img
    mogrify -crop "+100+105" $img
done


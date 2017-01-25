#!/bin/bash
list=`ls | grep .JPG` 
for img in $list; do
    echo "cropping $img"
    #mogrify -crop 1920x1080+900+1000
    mogrify -crop 1000x800+1000+1400 $img   
done


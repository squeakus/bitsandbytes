#!/bin/bash
list=`ls | grep .jpg` 
for img in $list; do
    echo "cropping $img"
    mogrify -crop 400x300+1400+1000 $img   
done


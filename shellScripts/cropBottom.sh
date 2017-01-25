#!/bin/bash
list=`ls | grep .jpg` 
for img in $list; do
    echo "cropping $img"
    mogrify -crop 2046x1500+0+0 $img   
done


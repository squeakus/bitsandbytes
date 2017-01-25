#!/bin/bash
list=`ls | grep .jpg` 
for img in $list; do
    echo "cropping $img"
    mogrify -crop 940x740-50+60 $img   
done


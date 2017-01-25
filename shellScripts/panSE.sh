#!/bin/bash
list=`ls | grep .jpg` 
offsetX=500
offsetY=400
for img in $list; do
    if [ $offsetX -lt 700 ]; then
	let offsetX=offsetX+1
	let offsetY=offsetY+1
    fi
    echo "cropping $img xOff $offsetX yOff $offsetY"
    mogrify -crop 800x600+$offsetX+$offsetY $img   
done


#!/bin/bash
list=`ls | grep .jpg` 
offsetX=1100
offsetY=000

for img in $list; do
    #if [ $offsetX -lt 700 ]; then
	#let offsetX=offsetX+1
	let offsetY=offsetY+1
    #fi
    echo "cropping $img xOff $offsetX yOff $offsetY"
    mogrify -crop 800x600+$offsetX+$offsetY $img   
done


#!/bin/bash
list=`ls | grep .jpg` 
width=3072
height=2304
offsetX=0
offsetY=0
for img in $list; do
    if [ $width -gt 800 ]; then 
	let width=width-4
	let offsetX=offsetX+2
	let height=height-3
	let mod=$offsetX%4

	if  [ $mod = 0 ]; then
    	    let offsetY=offsetY+1
	else
            let offsetY=offsetY+2
	fi
    fi
    echo "cropping $img width $width height $height x $offsetX y $offsetY"
    mogrify -crop $width'x'$height+$offsetX+$offsetY $img   
done


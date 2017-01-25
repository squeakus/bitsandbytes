#!/bin/bash
for file in front*.mesh; do 
    filename=`echo $file | sed s/front/run9front/`
    echo "converting $file to $filename"
    #`convert $file $filename`
done 

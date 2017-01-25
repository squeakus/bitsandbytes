#!/bin/bash
for file in *.[jJ][pP][gG]; do 
    filename=`echo $file | sed s/.jpg/.png/`
    echo "converting $file to $filename"
    `convert $file $filename`
done 

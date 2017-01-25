#!/bin/bash
for file in *.[pP][pP][mM]; do 
    filename=`echo $file | sed s/.ppm/.png/`
    echo "converting $file to $filename"
    #`convert $file $filename`
done 

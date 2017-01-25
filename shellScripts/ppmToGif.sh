#!/bin/bash
for file in *.[pP][pP][mM]; do 
    filename=`echo $file | sed s/.ppm/.gif/`
    echo "converting $file to $filename"
    #`convert $file $filename`
done 

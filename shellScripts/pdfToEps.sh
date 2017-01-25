#!/bin/bash
for file in *.[pP][dD][fF]; do 
    filename=`echo $file | sed s/.pdf/.eps/`
    echo "converting $file to $filename"
    `convert $file $filename`
done 

#!/bin/bash
for file in *.[eE][pP][sS]; do 
    filename=`echo $file | sed s/.eps/.jpg/`
    echo "converting $file to $filename"
    `convert $file $filename`
done 

#!/bin/bash

for filename in *.laz;do
    newname=`echo $filename | sed s/laz/las/`
    echo "Uncompressing $filename to $newname"
    las2las -i $filename -o $newname

done


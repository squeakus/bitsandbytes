#!/bin/bash

for file in *.hgt; do
    filename=`echo $file | sed s/hgt/aaigrid/`
    echo "gdal_translate -of AAIGrid $file $filename"
    gdal_translate -of AAIGrid $file $filename
done

rm -rf *.xml *.prj

#!/bin/bash
list=`ls | grep .tif`
for img in $list; do
    filename=`echo $img | sed s/tif/jpg/`
    echo "copying exif from $img to $filename"
    exiftool -TagsFromFile $img $filename
done


#!/bin/bash
if [ $# -lt 1 ];
then
    echo "please specify folder: tifconverter.sh <foldername>"
    exit 0
fi


destFolder=$1
echo "destination: $destFolder"
if [ -d "$destFolder" ]; then
    echo "FOLDER ALREADY EXISTS!"
    exit
fi
mkdir $destFolder


echo "copying tif files to $1"
for file in *.tif; do
    filename=`echo $file | sed s/tif/jpg/`
    echo "converting $file to $filename and copying to: $1"
    convert $file $filename
    exiftool -TagsFromFile $file $filename
    mv $filename $1
    rm -rf *.jpg_original
done

#!/bin/bash

if [ ! -d "$DIRECTORY" ]; then
    mkdir blackimages
fi

for image_file in *.jpg
do
    filesize=$(stat -c%s "$image_file")

    if [ $filesize -lt 100000 ]; then
        echo removing: $image_file
        mv $image_file ./blackimages
    fi
done

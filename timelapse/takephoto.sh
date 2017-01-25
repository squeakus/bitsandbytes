#!/bin/bash

var=`printf "img%04d.jpg" 1`
gphoto2 --auto-detect
gphoto2 --capture-image-and-download --force-overwrite
mv capt0000.jpg $var
echo "moving to $var image"


#!/bin/bash
echo "converting gifs to mp4"
for file in *.gif; do 
    filename=`echo $file | sed s/gif/mp4/`
    echo "converting $file to $filename"
    ffmpeg -f gif -i $file $filename
done 

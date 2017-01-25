#!/bin/bash
list=`ls | grep .JPG` 
for img in $list; do
    filename=${img%.*}
    echo $filename
    #convert "$filename.png" "$filename.png"
	
done


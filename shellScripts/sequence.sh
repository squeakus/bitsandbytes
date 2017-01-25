#!/bin/bash
COUNTER=0 
name="img"
for file in *.[jJ][pP][gG]; do 
    let COUNTER=COUNTER+1
    filename="$name`printf %04d $COUNTER`"
    if [ "$file" = "$filename.jpg" ]; then
	echo "filenames are identical!" 
    else
	echo "Renaming $file as $filename.jpg"
	#`mv $file $filename.jpg`
    fi
done 

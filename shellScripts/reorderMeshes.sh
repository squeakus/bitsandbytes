#!/bin/bash
COUNTER=0 
name="back."
for file in *back*.mesh; do 
    let COUNTER=COUNTER+1
    filename="$name$COUNTER.mesh"
    if [ "$file" = "$filename" ]; then
	echo "filenames are identical!" 
    else
	echo "Renaming $file as $filename"
	`mv $file $filename`
    fi
done 

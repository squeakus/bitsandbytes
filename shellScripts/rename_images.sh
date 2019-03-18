#!/bin/bash
COUNTER=0
name="beeright"
for file in *.jpg; do
    let COUNTER=COUNTER+1
    filename="$name`printf %05d $COUNTER`.jpg"
    if [ "$file" = "$filename" ]; then
        echo "filenames are identical!" 
    else
        echo "Renaming $file as $filename"
        mv $file $filename
    fi
done

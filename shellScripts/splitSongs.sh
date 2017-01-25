#!/bin/bash

counter=0
name="DrunkardsWalk"

# take the mp3s and split em into 5 mins
for file in *.mp3; do
    echo "splitting file $file"
    `mp3splt -t 5.00 -o @f_@n3 -a -d split $file`
done

# if [ -d "ordered" ]; then
#     echo "FOLDER ALREADY EXISTS!"
#     exit
# fi
# mkdir ordered

# for split in ./split/*.mp3; do
#     let counter=counter+1
#     filename="$name`printf %04d $counter`"
#     echo "moving $split to ./ordered/$filename"
# done
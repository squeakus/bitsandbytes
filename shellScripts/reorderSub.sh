#!/bin/bash
counter=0
list=`find . -type f -name "front*.mesh"`
for file in $list; do
    let counter=counter+1
    filename="front.$counter.mesh"
    echo "converting $file to $filename"
    `cp $file $filename`
done 

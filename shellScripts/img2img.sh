#!/bin/bash
if [ $# -lt 2 ];
then
    echo "not enough args"
    exit 0
fi

for filename in *.$1;do
    newname=`echo $filename | sed 's/'$1'/'$2'/'`
    echo "moving $filename to $newname"
    convert $filename $newname
done

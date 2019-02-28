#!/bin/bash
if [ $# -lt 1 ];
then
    echo "not enough args"
    exit 0
fi

#check if folder exists else output to STDERR
if [ -d $1 ];
then
   echo "copying files to $1"
else
   echo "The directory does not exist or is mounted incorrectly"
   exit 1
fi

for labelname in *.xml;do
    imagename=`echo $labelname | sed 's/'.xml'/'.jpg'/'`
    echo "copying $labelname and $imagename to $1"
    cp $labelname $1
    cp $imagename $1
done

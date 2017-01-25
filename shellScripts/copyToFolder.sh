#!/bin/bash
COUNTER=0
mod=2 
write=0

if [ $# -lt 1 ];
then
    echo "please specify folder: copytofolder.sh <foldername>"
    exit 0
fi


destFolder=$1
echo "destination: $destFolder"
if [ -d "$destFolder" ]; then
    echo "FOLDER ALREADY EXISTS!"
    exit
fi
mkdir $destFolder

list=`ls | grep .jpg` 
for img in $list; do
    let write=COUNTER%mod
    let COUNTER=COUNTER+1
    if [ $write = 0 ]; then
	echo "copying $img to $destFolder"
	cp $img $destFolder
    fi
done

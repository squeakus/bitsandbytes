#!/bin/bash
if [ $# -lt 2 ];
then
    echo "not enough args"
    exit 0
fi

for DIRNAME in *;do
    newname=`echo $DIRNAME | sed 's/'$1'/'$2'/'`
    echo ${DIRNAME} $newname
    mv ${DIRNAME} $newname
done

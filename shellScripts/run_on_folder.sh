#!/bin/bash

if [ $# -lt 1 ];
then
    echo "please specify folder: run_on_folder.sh <foldername>"
    exit 0
fi


scanFolder=$1
echo "processing: $scanFolder"

find $scanFolder -maxdepth 2 -type d  -exec python ebr45.py '{}' \;

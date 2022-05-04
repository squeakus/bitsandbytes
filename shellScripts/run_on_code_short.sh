#!/bin/bash

if [ $# -lt 1 ];
then
    echo "please specify folder: run_on_folder.sh <foldername>"
    exit 0
fi


scanFolder=$1

for d in $scanFolder*/; do
	echo "processing: $d"
	python blindsight_checker.py $d
done


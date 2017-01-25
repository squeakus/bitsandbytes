#!/bin/bash

for filename in *.las; do
	txtname=`basename $filename .las`.txt
	echo "converting $filename to $txtname"
	las2txt -i $filename -o $txtname
done

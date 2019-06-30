#!/bin/bash
mkdir -p alllabels
mkdir -p all
# images=`find . -name ! -path ./all "*.jpg"`
# for f in $images; 
# do 
# 	newname=`echo $f | sed 's/\///' | sed 's/.//' | sed 's/\//_/'`
# 	cp "$f" "all/$newname"; 
# done

labels=`find . -name "*.xml" ! -path ./alllabels`
for f in $labels; 
do 
	newname=`echo $f | sed 's/\///' | sed 's/.//' | sed 's/\//_/'`
	echo "copying $f to alllabels/$newname"
	cp "$f" "alllabels/$newname"; 
done


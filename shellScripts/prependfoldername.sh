#!/bin/bash
images=`find . -name "*.jpg"`
for f in $images; 
do 
	newname=`echo $f | sed 's/\///' | sed 's/.//' | sed 's/\//_/'`
	echo $newname
	echo "$f" "all/$newname"; 
done

#rename 's/^/wide/' *

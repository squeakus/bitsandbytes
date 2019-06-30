#!/bin/bash
mkdir -p alllabels
mkdir -p all
images=`find . -name "*.jpg" ! -path ./all`

for f in $images; 
do 
	newname=`echo $f | sed 's/\///' | sed 's/.//' | sed 's/\//_/'`
	echo "copying $newname"
	cp "$f" "all/$newname"; 
done

labels=`find . -name "*.xml" ! -path ./alllabels`
for f in $labels; 
do 
	newname=`echo $f | sed 's/\///' | sed 's/.//' | sed 's/\//_/'`
	echo "copying $f to alllabels/$newname"
	cp "$f" "alllabels/$newname";
	fname=$(basename "${f%.*}")
	newfname=${newname%.*}
	echo "sed -i 's/$fname/$newfname/g' alllabels/$newname"
	sed -i "s/$fname/$newfname/g" "alllabels/$newname"
done


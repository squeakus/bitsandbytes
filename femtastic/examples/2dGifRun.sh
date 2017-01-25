#!/bin/bash
for file in *.[eE][pP][sS]; do 
    filename=`echo $file | sed s/.eps/.jpg/`
    echo "converting $file to $filename"
    `convert $file $filename`
done 

list=`ls | grep .jpg` 
for img in $list; do
    echo "tidying up $img"
    mogrify -rotate "+45>" $img
    mogrify -crop "-190-190" $img
    mogrify -crop "+190+190" $img
done
`convert *.jpg run.gif`
for file in *.[eE][pP][sS]; do
    filename=`echo $file | sed s/.eps/.jpg/`
    rm $file
    rm $filename
done


#!/bin/bash
list=`ls | grep .gif` 
for img in $list; do
    echo "tidying up $img"
    mogrify -crop "-50-50" $img
    mogrify -crop "+50+50" $img
done


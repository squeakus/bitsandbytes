#!/bin/bash

for img in *.jpg; do 
        brightfloat=`convert $img -colorspace Gray -format "%[mean]" info:`
        brightint=${brightfloat%.*}
	echo "$img brightness $brightint"
        if [ $brightint -lt "10000" ];
        then
                echo "$img its dark now"
                break
        fi
done


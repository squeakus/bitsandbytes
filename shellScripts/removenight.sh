#!/bin/bash

for img in *.jpg; do
    greyval=`convert $img -format "%[mean]" info:`
    # be careful, this always rounds down!
    greyint=${greyval%.*}
    echo "image $img has greyval $greyint"
    if [ $greyint -lt 22000 ]; then
        echo "moving $img"
        mv $img ./night
    fi
done


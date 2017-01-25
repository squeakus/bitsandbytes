#!/bin/bash

for filename in *.laz; do
    las2txt $filename -o xyz.txt
    txt2las xyz.txt -o cleaned.las
    las2las  cleaned.las  --a_srs "+proj=tmerc +lat_0=53.5 +lon_0=-8 +k=1.000035 +x_0=200000 +y_0=250000 +ellps=mod_airy +towgs84=482.5,-130.6,564.6,-1.042,-0.214,-0.631,8.15 +units=m +no_defs " -o assigned.las

    lasname=`echo $filename | sed s/laz/las/`
    newname="new"$lasname
    las2las assigned.las --t_srs EPSG:3587 -o $newname
    laszip $newname
    rm xyz.txt cleaned.las assigned.las $newname
done

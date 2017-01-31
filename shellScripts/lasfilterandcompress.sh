#!/bin/bash
mkdir filtered
for filename in *.laz; do
    CloudCompare -SILENT -NO_TIMESTAMP -AUTO_SAVE OFF -C_EXPORT_FMT LAS -O $filename -SOR 6 10 -CROP 0:0:0:9999999:9999999:9999999 -SAVE_CLOUDS

done

for filename in *.las;do
    newname=`echo $filename | sed s/las/laz/`
    echo "compressing $filename to $newname"
    las2las -i $filename -o $newname
    mv $newname filtered
    #rm $filename
done

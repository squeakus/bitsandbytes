#!/bin/bash

if [ $# -eq 0 ]
  then
    echo " usage: skin.sh <filename>"
else

cc=/usr/local/CloudCompare/CloudCompare
echo $cc -O $1 -C_EXPORT_FMT PLY -SS SPATIAL 0.05 
$cc -SILENT -O $1 -C_EXPORT_FMT PCD -SS SPATIAL 0.05 
subsample=`ls | grep SPATIAL`
echo moving $subsample
mv $subsample subsample.asc
meshlabserver -i ./subsample.asc -o ./meshed.ply -s ~/Jonathan/programs/pointcloud/poisson.mlx -om vf vn ff
fi

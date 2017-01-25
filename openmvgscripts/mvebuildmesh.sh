#!/bin/bash
# Convert the openMVG SfM scene to the MVE format
start=`date +%s`



if [ $# -lt 1 ];
then
    echo "please specify folder: mvebuildmesh <foldername>"
    exit 0
fi
openMVG_main_openMVG2MVE2 -i $1/sfm_data.bin -o $1

directory=$1/MVE
resolution=2

# MVE
dmrecon -s$resolution $directory
scene2pset -ddepth-L$resolution -iundist-L$resolution -n -s -c $directory $directory/OUTPUT.ply
fssrecon $directory/OUTPUT.ply $directory/OUTPUT_MESH.ply
meshclean $directory/OUTPUT_MESH.ply $directory/OUTPUT_MESH_CLEAN.ply
end=`date +%s`
runtime=$((end-start))
echo "Time taken: $runtime"

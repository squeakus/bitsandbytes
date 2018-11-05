#!/bin/bash
if [ $# -lt 1 ];
then
    echo "please specify folder: colmap.sh <foldername>"
    exit 0
fi

print_time () {
    current=`date +%s`
    secs=$((current-$1))
    echo ""
    printf 'time taken: %dh:%dm:%ds\n' $(($secs/3600)) $(($secs%3600/60)) $(($secs%60))

}

start=`date +%s`
TARGET=$1
printf "Deleting existing database"
rm $TARGET/$TARGET.db

printf "\nRunning feature extractor"
# To run on CPU use --SiftExtraction.use_gpu 0 
colmap feature_extractor  --database_path $TARGET/$TARGET.db --image_path $TARGET/ > $TARGET".log"
print_time "$start"

printf "\nRunning feature matcher"
# To run on CPU use --SiftMatching.use_gpu 0
colmap exhaustive_matcher  --database_path=$TARGET/$TARGET.db >> $TARGET".log"
print_time "$current"

mkdir -p $TARGET/sparse
printf "\nRunning sparse reconstruction"
colmap mapper --database_path $TARGET/$TARGET.db --image_path $TARGET/ --output_path $TARGET/sparse/ >> $TARGET".log"
print_time "$current"

colmap model_converter  --input_path $TARGET/sparse/0 --output_path $TARGET.ply --output_type ply

mkdir -p $TARGET/dense
printf "\nUndistorting images"
colmap image_undistorter \
    --image_path $TARGET/ \
    --input_path $TARGET/sparse/0 \
    --output_path $TARGET/dense \
    --output_type COLMAP \
    --max_image_size 2000 >> $TARGET".log"
print_time "$current"

printf "\nStereo matching"
colmap patch_match_stereo \
    --workspace_path $TARGET/dense \
    --workspace_format COLMAP \
    --PatchMatchStereo.geom_consistency true >> $TARGET".log"
print_time "$current"

printf "\nFusing Stereo"
colmap stereo_fusion \
    --workspace_path $TARGET/dense \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path $TARGET/dense/fused.ply >> $TARGET".log"
print_time "$current"

printf "\nPoisson Meshing"
colmap poisson_mesher \
    --input_path $TARGET/dense/fused.ply \
    --output_path $TARGET/dense/meshed-poisson.ply >> $TARGET".log"
print_time "$current"

printf "\nDelaunay Meshing"
colmap delaunay_mesher \
    --input_path $TARGET/dense/fused.ply \
    --output_path $TARGET/dense/meshed-delaunay.ply >> $TARGET".log"
print_time "$current"

printf "Finished!"
print_time "$start"


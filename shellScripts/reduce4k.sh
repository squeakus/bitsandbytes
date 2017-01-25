#!/bin/bash

for vid in *.MP4;do
   mencoder $vid -ovc x264 -x264encopts subq=6:partitions=all:8x8dct:me=umh:frameref=5:bframes=3:b_pyramid=normal:weight_b -vf scale=1920:1080 -oac copy -o reduced$vid

done

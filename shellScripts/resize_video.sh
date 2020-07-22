#!/bin/bash
list=`ls | grep '.mp4\|.MP4\|.avi\|.AVI'`
for vid in $list; do
	echo "resizing $vid to 1080p_$vid"
	mencoder $vid -ovc lavc -lavcopts vcodec=mp4 -vop scale=1920:1080 -oac copy -o 1080p_$vid
done
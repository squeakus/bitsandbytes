#!/bin/bash
mkdir -p /home/pi/images
while true; do
	IMGNAME=/home/pi/images/img$(date +"%Y%m%d%H%M%S").jpg
	echo "capturing image $IMGNAME"
	raspistill -o $IMGNAME
	sleep 30
done

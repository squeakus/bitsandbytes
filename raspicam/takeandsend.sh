#!/bin/bash
mkdir -p /home/pi/images
while true; do
	IMGNAME=/home/pi/images/img$(date +"%Y%m%d%H%M%S").jpg
	echo "capturing image $IMGNAME"
	raspistill -o $IMGNAME
	raspistill -w 640 -h 480 -q 100 -o $IMGNAME
	scp $IMGNAME jonathan@192.168.1.197:images
	sleep 10
	rm -f $IMGNAME

done

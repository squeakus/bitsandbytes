#!/bin/bash
mkdir -p /home/pi/images
while true; do
	IMGNAME=/home/pi/images/img$(date +"%Y%m%d%H%M%S").jpg
	echo "capturing image $IMGNAME"
	raspistill -w 640 -h 480 -q 100 -o $IMGNAME

            brightfloat=`convert $IMGNAME -colorspace Gray -format "%[mean]" info:`
        brightint=${brightfloat%.*}
        if [ $brightint -lt "10000" ]; then
            echo ""$IMGNAME its dark now
            sleep 600
        else
            scp $IMGNAME jonathan@192.168.1.197:images
        sleep 5 
        fi


	rm -f $IMGNAME

done

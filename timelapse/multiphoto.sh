#!/bin/bash

mkdir /home/pi/images

for i in {1..1000}

do
	var=`printf "/home/pi/images/img%04d.jpg" $i`
	gphoto2 --auto-detect
	gphoto2 --capture-image-and-download --force-overwrite
	mogrify -resize 1920x1080 capt0000.jpg
	mv capt0000.jpg $var
	echo "moving to $var image"
	#sleep 10
	brightfloat=`convert $var -colorspace Gray -format "%[mean]" info:`
	brightint=${brightfloat%.*}
	if [ $brightint -lt "90" ];
	then 
		echo "" its dark now
		break
	fi
done

DATE=`date +%Y-%m-%d`-sunset
mv /home/pi/images /home/pi/$DATE
cd /home/pi/$DATE
ls | grep .jpg > frames.txt
mencoder mf://@frames.txt -mf w=1920:h=1080:fps=20:type=jpg -ovc x264 -x264encopts subq=6:partitions=all:8x8dct:me=umh:frameref=5:bframes=3:b_pyramid=normal:weight_b -o a.avi
VIDNAME=`date +%Y-%m-%d`.avi
mv a.avi $VIDNAME
scp $VIDNAME jonathan@macbrew.ucd.ie:Dropbox/wedding

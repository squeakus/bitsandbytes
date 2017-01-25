#!/bin/bash
cd /mnt/usb
DATE=$(date +"%Y-%m-%d-%H-%M")
tar -cvpf $DATE.tar *.jpg
sshpass -p "" scp $DATE.tar jonathan@jonserver.ucd.ie:
ls | grep .jpg > frames.txt
COUNT=`ls | grep .jpg | wc -l`
mencoder mf://@frames.txt -mf w=800:h=600:fps=12:type=jpg -ovc xvid -ovc x264 -x264encopts bitrate=3000:pass=1:nr=2000 -o a.avi
uuencode a.avi a.avi | mailx -s "video for $DATE image count $COUNT" jonathanbyrn@gmail.com
rm *.tar *.avi *.log* *.txt

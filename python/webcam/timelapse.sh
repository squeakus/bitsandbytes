#!/bin/bash
sleep 30
currentdate=`/bin/date`
/usr/bin/fswebcam -r 1280x960 -S 15 -d /dev/video0 test.jpg

uuencode test.jpg test.jpg | mail -s "now up and running:  $currentdate" jonathanbyrn@gmail.com
sudo mount -t vfat /dev/sda1 /mnt/usb
/usr/bin/fswebcam -r 1280x960 -S 15 -d /dev/video0 -l 300 /mnt/usb/img-%Y-%m-%d--%H-%M-%S.jpg

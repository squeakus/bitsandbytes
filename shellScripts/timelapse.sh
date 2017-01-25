#!/bin/bash
#remove previous gphoto save
`rm capt0000.jpg > /dev/null`
MAXLOOPS=5
INTERVAL=5
COUNTER=0

while [  $COUNTER -lt $MAXLOOPS ]; do
  echo Image $COUNTER:
  echo $COUNTER > counter.txt
  
  #take photo and check its okay
  `gphoto2 --capture-image-and-download > /dev/null` 
  `echo exiv2 capt0000.jpg > /dev/null`
   RETVAL=$?
  [ $RETVAL -eq 0 ] && echo successfully saved image
  [ $RETVAL -ne 0 ] && `echo "Image $COUNTER has failed to save"|
                       mailx -s "Image capture failed" jonathanbyrn@gmail.com`

  #copy file and resize
  original="./original/img`printf %04d $COUNTER`.jpg"
  resized="./resized/resize`printf %04d $COUNTER`.jpg"
  `mv capt0000.jpg $original`
  `cp $original $resized`
  `mogrify -resize 1920x1080 $resized`
  
   #mogrify -crop 1920x1080+0+0 test.jpg - added for testing zoom

  #sleep until next shot
  sleep $INTERVAL
  let COUNTER=COUNTER+1
done
echo Script Finished

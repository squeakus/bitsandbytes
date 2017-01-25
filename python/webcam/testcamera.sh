#!/bin/bash
/usr/bin/fswebcam -r 2000x2000 -S 15 -d /dev/video0 test.jpg
#uuencode test.jpg test.jpg | mail -s "testing camera" jonathanbyrn@gmail.com

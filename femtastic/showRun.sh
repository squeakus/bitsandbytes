#!/bin/bash
for file in *.[pP][pP][mM]; do
    filejpg=`echo $file | sed s/.ppm/.jpg/`
    echo "converting $file to $filejpg"
    `convert $file $filejpg`
    rm $file
done
#ffmpeg -r 10 -b 1800 -i img%04d.jpg run.mp4
#mencoder mf://*.jpg -mf w=800:h=600:fps=25:type=jpg -ovc xvid -xvidencopts pass=1:trellis:bitrate=800 -o a.divx
#for file in *.jpg; do
#    rm $file
#done
#rm divx2pass.log
gnome-open run.mp4


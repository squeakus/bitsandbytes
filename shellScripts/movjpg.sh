#!/bin/bash
ls | grep .jpg > frames.txt
mencoder mf://@frames.txt -mf w=800:h=600:fps=60:type=jpg -ovc xvid -xvidencopts pass=1:trellis:bitrate=800 -o a.divx
#open a.divx

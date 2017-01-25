#!/bin/bash
ls | grep .png > frames.txt
mencoder mf://@frames.txt -mf w=800:h=600:fps=15:type=png -ovc xvid -xvidencopts pass=1:trellis:bitrate=800 -o a.divx
#open a.divx

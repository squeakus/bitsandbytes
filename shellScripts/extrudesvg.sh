#!/bin/bash
echo "converting $1 to stl"
inkscape -E intermediate.eps $1
outfile=`echo $1 | sed s/.svg/.dxf/`
pstoedit -dt -f dxf:-polyaslines intermediate.eps $outfile
echo $outfile
rm intermediate.eps
openscad $outfile

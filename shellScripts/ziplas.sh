#!/bin/bash
lascmd="/Users/jonathan/LAStools/bin/laszip"
for lasfile in *.las
do
   echo "zipping file $lasfile"
   $lascmd $lasfile
done


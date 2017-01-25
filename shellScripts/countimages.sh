#!/bin/bash

localpath="/home/jonathan/Pictures/canonEOS"
previous=`cat $localpath/previous.txt`
total=`ls $localpath/original | grep .jpg | wc -w`
let taken=$total-$previous
currentdate=`date`
`echo "story! photos taken today: $taken total: $total"|
 mailx -s "photo report for $currentdate" jonathanbyrn@gmail.com`

#echo $total > previous.txt
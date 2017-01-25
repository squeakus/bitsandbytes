#!/bin/bash
count=`ls | grep .mesh -c`
let count=count-1
echo 'FOUND:'$count
cmd='ffmedit xxx -a 1 '$count
$cmd
`convert xxx*.ppm run.gif`
for file in *.ppm; do
    rm $file
done
for file in *.mesh; do
    rm $file
done
#gnome-open run.gif

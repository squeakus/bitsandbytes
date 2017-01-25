#!/bin/bash
count=`ls | grep front. -c`
let count=count-1
echo 'FOUND:'$count
cmd='ffmedit front -a 1 '$count
$cmd
`convert front*.ppm runF.gif`
for file in *.ppm; do
    rm $file
done
for file in front*; do
    rm $file
done
count=`ls | grep back. -c`

let count=count-1
echo 'FOUND:'$count
cmd='ffmedit back -a 1 '$count
$cmd
`convert back*.ppm runB.gif`
for file in *.ppm; do
    rm $file
done
for file in back*; do
    rm $file
done

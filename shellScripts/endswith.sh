#!/bin/bash
# $? holds the exit status
echo"$1" |grep '\.tex$'
if [ $? -eq 0 ];then
    echo "Ends with tex"
    newvar=${1%.tex}
    echo "eereplaced $newvar"
fi

echo $1 |grep '\.$'
if [ $? -eq 0 ];then
    newvar=${1%.}
    echo "$newvar"
fi

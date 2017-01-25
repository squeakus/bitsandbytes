#!/bin/bash

function checklist()
{
    find=$1
    
    found=1 #zero is true in bash 
    while read line
    do
	if [[ "$line" == *$find* ]]; then
	    var=$(echo $line | awk '{print $2}')
	    echo "found $find $var"
	    found=0
	fi
#  case "$line" in
#      */dev/sda1*) echo "found!!!!" ;;
#  esac
    done < devlist.txt

    if [ $found -eq 1 ]; then #if it is not found
	echo "$find -1" >> devlist.txt
    fi
}

alwaystrue() { return 0; }
moo=0
if $moo; then echo "MOOO" 
fi

#list of all partitions and check them
PARTITIONS=`df -h | grep dev/sd |awk '{print $1}'`

for partition in $PARTITIONS
do
    checklist $partition
done
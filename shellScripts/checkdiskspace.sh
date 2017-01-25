#!/bin/bash

function check_partition {
    echo "arg 1 $1"
    # grep/awk the percentage used
    PREV_FILE="prev_space.txt"
    SPACE=`df -h | grep sda | awk '{print $5}'`
    SPACE=${SPACE%'%'}

#check that noone has deleted the file
    if [ ! -e $PREV_FILE ]; then
	echo "0" > $PREV_FILE
    fi

# get prev value from the file
    PREV_SPACE=`cat $PREV_FILE`
    MOD=$(expr $SPACE % 10)
    let "ROUNDED = $SPACE - $MOD"

# check for increase
    if [ $ROUNDED -gt $PREV_SPACE ]; then
	echo "server $HOSTNAME now $ROUNDED% full"
	echo "$ROUNDED" > $PREV_FILE
    fi

# check if above 90%
    if [ $SPACE -gt 90 ]; then
	echo "over 90! running out of space on $HOSTNAME"
    else
	echo "under 90"
    fi
}

check_partition sda1
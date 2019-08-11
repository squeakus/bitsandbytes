#!/bin/bash

CURRENT=$(df / | grep / | awk '{ print $5}' | sed 's/%//g')
THRESHOLD=60

if [ "$CURRENT" -gt "$THRESHOLD" ] ; then
   printf "Subject: disk space low\n\nCurrent disk space: $CURRENT%" | ssmtp jonathanbyrn@gmail.com
fi

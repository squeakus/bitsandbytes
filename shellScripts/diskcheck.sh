#!/bin/bash
# Run in crontab with:
#37 11 * * * /home/pi/diskcheck.sh > /home/pi/crontab.log 2>&1

CURRENT=$(df / | grep / | awk '{ print $5}' | sed 's/%//g')
THRESHOLD=60

if [ "$CURRENT" -gt "$THRESHOLD" ] ; then
   printf "Subject: disk space low\n\nCurrent disk space: $CURRENT%%" | /usr/sbin/ssmtp jonathanbyrn@gmail.com
fi

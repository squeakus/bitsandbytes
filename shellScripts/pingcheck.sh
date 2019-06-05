#!/bin/bash

ping_targets="raspberrypi.local moo.blah"
failed_hosts=""

for i in $ping_targets
do
   ping -c 1 $i > /dev/null
   if [ $? -ne 0 ]; then
      if [ "$failed_hosts" == "" ]; then
         failed_hosts="$i"
      else
         failed_hosts="$failed_hosts, $i"
      fi
   fi
done

if [ "$failed_hosts" != "" ]; then
   printf "Subject: server down\n\nThe following hosts are down: $failed_hosts" | ssmtp jonathanbyrn@gmail.com
fi

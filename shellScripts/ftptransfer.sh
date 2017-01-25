#!/bin/sh
USER=ncra@johnmarkswafford.com
PASSWD=52215lab
ftp -n ftp.johnmarkswafford.com <<SCRIPT
user $USER $PASSWD
cd ncrabackup
lcd ~
put pylon1.zip
quit
SCRIPT


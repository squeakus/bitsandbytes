#!/bin/bash

# Modified from http://www.whoopis.com/howtos/rsync-hfs-howto/
# Space-separated list of directories to back up; edit as needed
FILES="f1pl2009.latest.sql ncra.latest.sql ncra.latest.tgz ge.org.latest.tgz svn.latest.tgz cgi.latest.tgz fmcc.tgz fmcc.latest.sql"

# Options to pass to rsync; edit as needed
# "--delete" = destructive backup 
#   (i.e. if you delete a local file, it will be deleted 
#   on the server at next backup; keeps local+backup synchronized)
# "--update" = update only (don't overwrite newer versions of files)
OPTS="--update"

# Backup destination. In this case, it is another hard disk on the same machine.
# Incidentally, it is DOS-formatted, irrelevant here.
# If you wish to back up to a server via ssh, change the line to something like
# BACKUPDIR="remoteusername@someserver.something:/path/to/backup/destination"
LOCALBACKUPDIR="/Volumes/KINGSTON/NRAbackup/"
NETWORKBACKUPDIR="/Volumes/galapagosBackup/"

# ignore Mac droppings
EXCLUDES="--exclude .DS_Store --exclude .Trash --exclude Cache --exclude Caches"

# Build the actual command
# NOTE the specific path to the "special" version of rsync
LOCALBACKUP="/usr/bin/rsync -a $OPTS $EXCLUDES $FILES $LOCALBACKUPDIR"
NETWORKBACKUP="/usr/bin/rsync -a $OPTS $EXCLUDES $FILES $NETWORKBACKUPDIR"
# Informative output
date
echo Starting do not close this window until it is finished.
rm $FILES

echo Dumping out Databases
/usr/local/mysql/bin/mysqldump -uroot -p52215lab f1pl2009 > f1pl2009.latest.sql
/usr/local/mysql/bin/mysqldump -uroot -p52215lab ncra > ncra.latest.sql
/usr/local/mysql/bin/mysqldump -uroot -p52215lab fmcc > fmcc.latest.sql

echo Zipping files
tar czPf fmcc.tgz /Library/WebServer/Documents/fmcc
tar czPf ncra.latest.tgz /Library/WebServer/Documents/ncra
tar czPf ge.org.latest.tgz /Library/WebServer/Documents/ge.org
tar czPf svn.latest.tgz /usr/local/svn
tar czPf cgi.latest.tgz /Library/WebServer/CGI-Executables
perl /Library/WebServer/CGI-Executables/getnews2rss.cgi Qcount=10  > /Library/WebServer/Documents/ncra/ncra.xml

echo Copying files to local backup:

#check if folder exists else output to STDERR
if [ -d "$LOCALBACKUPDIR"]; then
   echo $LOCALBACKUP
   $LOCALBACKUP
else
   echo "The directory does not exist or is mounted incorrectly" 1>&2
fi

#check if network folder exists then copy
echo Copying files to network backup:
if [ -d "$NETWORKBACKUPDIR"]; then
   echo $NETWORKBACKUP
   $NETWORKBACKUP
else
   echo "The directory does not exist or is mounted incorrectly" 1>&2
fi
echo Done.
date
# the end.

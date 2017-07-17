#!/bin/bash

# Create three files: allfileshas.txt, bigobjects.txt and bigtosmall.txt
# allfileshas.txt: list of all objects in the repository history
# bigobjects.txt: list big objects
# bigtosmall.txt: list biggest objects from biggest to smallest

if [ -f allfileshas.txt ]; then
	rm allfileshas.txt bigobjects.txt bigtosmall.txt
fi
git rev-list --objects --all | sort -k 2 > allfileshas.txt
git gc && git verify-pack -v .git/objects/pack/pack-*.idx | egrep "^\w+ blob\W+[0-9]+ [0-9]+ [0-9]+$" | sort -k 3 -n -r > bigobjects.txt
for SHA in `cut -f 1 -d\  < bigobjects.txt`; do
echo $(grep $SHA bigobjects.txt) $(grep $SHA allfileshas.txt) | awk '{print $1,$3,$7}' >> bigtosmall.txt
done;

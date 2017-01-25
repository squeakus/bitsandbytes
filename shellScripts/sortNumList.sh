#!/bin/bash                                                                    
if [ -f "mylist.txt" ]
then
rm mylist.txt
fi
while read list
do
echo  $list >> mylist.txt
done
cat mylist.txt | sort -nr -k2 -o mylist.txt
head -n 3 mylist.txt

#!/bin/bash                                                                    
if [ -f "mylist.txt" ]
then
rm mylist.txt
fi
while read list
do
echo  $list >> mylist.txt
sort mylist.txt -o mylist.txt
done
cat mylist.txt

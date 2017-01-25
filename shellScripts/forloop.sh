#/bin/bash
echo "Bash version ${BASH_VERSION}..."

for i in 1 2 3 4 5
do
   echo "Welcome $i times"
done


for i in {1..5}
do
   echo "Welcome $i times"
done

for i in {0..10..2}
  do
     echo "Welcome $i times"
 done

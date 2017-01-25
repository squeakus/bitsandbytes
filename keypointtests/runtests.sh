#!/bin/bash

#echo `date`": starting sift60" > log.txt
#python comparedetectors.py richviewnadir60/ sift
echo `date`": starting sift50" >> log.txt
python comparedetectors.py richviewnadir50/ sift
#echo `date`": starting sift30" >> log.txt 
#python comparedetectors.py richviewnadir30/ sift
#echo `date`": starting surf60" >> log.txt 
#python comparedetectors.py richviewnadir60/ surf
#echo `date`": starting surf30" >> log.txt 
echo `date`": starting surf50" >> log.txt
python comparedetectors.py richviewnadir50/ surf
#python comparedetectors.py richviewnadir30/ surf
#echo `date`": starting orb60" >> log.txt 
#python comparedetectors.py richviewnadir60/ orb
#echo `date`": starting orb30" >> log.txt
echo `date`": starting orb50" >> log.txt
python comparedetectors.py richviewnadir50/ orb
#python comparedetectors.py richviewnadir30/ orb
#echo `date`": starting akaze60" >> log.txt
#python comparedetectors.py richviewnadir60/ akaze
#echo `date`": starting akaze30" >> log.txt
echo `date`": starting akaze50" >> log.txt
python comparedetectors.py richviewnadir50/ akaze 
#python comparedetectors.py richviewnadir30/ akaze
#echo `date`": starting brisk60" >> log.txt 
#python comparedetectors.py richviewnadir60/ brisk
echo `date`": starting brisk50" >> log.txt
python comparedetectors.py richviewnadir50/ brisk
#echo `date`": starting brisk30" >> log.txt 
#python comparedetectors.py richviewnadir30/ brisk


echo `date`": starting sift60" >> log.txt
python comparedetectors.py feb16nadir70m/ sift
echo `date`": starting sift50" >> log.txt
python comparedetectors.py feb16nadir50m/ sift
echo `date`": starting sift30" >> log.txt 
python comparedetectors.py feb16nadir30m/ sift
echo `date`": starting surf60" >> log.txt 
python comparedetectors.py feb16nadir70m/ surf
echo `date`": starting surf50" >> log.txt
python comparedetectors.py feb16nadir50m/ surf
echo `date`": starting surf30" >> log.txt 
python comparedetectors.py feb16nadir30m/ surf
echo `date`": starting orb60" >> log.txt 
python comparedetectors.py feb16nadir70m/ orb
echo `date`": starting orb50" >> log.txt
python comparedetectors.py feb16nadir50m/ orb
echo `date`": starting orb30" >> log.txt
python comparedetectors.py feb16nadir30m/ orb
echo `date`": starting akaze60" >> log.txt
python comparedetectors.py feb16nadir70m/ akaze
echo `date`": starting akaze50" >> log.txt
python comparedetectors.py feb16nadir50m/ akaze
echo `date`": starting akaze30" >> log.txt
python comparedetectors.py feb16nadir30m/ akaze
echo `date`": starting brisk60" >> log.txt 
python comparedetectors.py feb16nadir70m/ brisk
echo `date`": starting brisk50" >> log.txt
python comparedetectors.py feb16nadir50m/ brisk
echo `date`": starting brisk30" >> log.txt 
python comparedetectors.py feb16nadir30m/ brisk


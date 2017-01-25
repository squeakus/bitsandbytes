#!/usr/bin/python                                                               
import sys
from string import maketrans

alphabet = "abcdefghijklmnopqrstuvwxyz"

def shifttext(oldtext,shiftNum):
    oldtext = ''.join(oldtext)
    # shifts your alphabet by shiftNum
    alphaShift = alphabet[shiftNum:] + alphabet[:shiftNum]
    # makes a translation table from your alphabet to your shifted alphabet
    transTab = maketrans(alphabet + alphabet.upper(), alphaShift + alphaShift.upper())
    # store the newly translated line
    newtext = oldtext.translate(transTab)
    return newtext

text=[]

for line in sys.stdin:
     text +=line
sample =text[0:200]
resultsArray =[]
for shiftnum in range(0,26):
     textArray = shifttext(sample,shiftnum)
     found = 0
     for i in range(0,(len(textArray)-3)):
          if((textArray[i]=='t')and(textArray[i+1]=='h')and(textArray[i+2]=='a')and(textArray[i+3]=='t')):
               found+= 3;
          if((textArray[i]=='h')and(textArray[i+1]=='a')and(textArray[i+2]=='v')and(textArray[i+3]=='e')):
               found+= 3;
          if((textArray[i]=='w')and(textArray[i+1]=='i')and(textArray[i+2]=='t')and(textArray[i+3]=='h')):
               found+= 3;
          if((textArray[i]=='t')and(textArray[i+1]=='h')and(textArray[i+2]=='i')and(textArray[i+3]=='s')):
               found+= 3;
          if((textArray[i]=='f')and(textArray[i+1]=='r')and(textArray[i+2]=='o')and(textArray[i+3]=='m')):
               found+= 3;
          if((textArray[i]=='a')and(textArray[i+1]=='n')and(textArray[i+2]=='d')):
               found+= 2;
          if((textArray[i]=='t')and(textArray[i+1]=='h')and(textArray[i+2]=='e')):
               found+= 2;
          if((textArray[i]=='f')and(textArray[i+1]=='o')and(textArray[i+2]=='r')):
               found+= 2;
          if((textArray[i]=='n')and(textArray[i+1]=='o')and(textArray[i+2]=='t')):
               found+= 2;
	  if((textArray[i]=='b')and(textArray[i+1]=='e')):
               found+= 1;
	  if((textArray[i]=='t')and(textArray[i+1]=='o')):
               found+= 1;
	  if((textArray[i]=='o')and(textArray[i+1]=='f')):
               found+= 1;
	  if((textArray[i]=='i')and(textArray[i+1]=='n')):
               found+= 1;
	  if((textArray[i]=='i')and(textArray[i+1]=='t')):
               found+= 1;
     resultsArray.insert(shiftnum,found)
max =0

for result in resultsArray:
     if result > max:
          max =result
index = resultsArray.index(max)
fixed = shifttext(text,index)
fixed = ''.join(fixed)
fixed = fixed.rstrip("\n")
print fixed

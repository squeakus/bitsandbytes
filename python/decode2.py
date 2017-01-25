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
     for i in range(0,200):
         if textArray[i] == 'e':
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


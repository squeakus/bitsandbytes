#!/usr/bin/python                                                               
import sys

alphabet = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]

def shifttext(oldtext, shiftnum):
     oldtext = list(oldtext)
     newtext =[]
     for char in oldtext:
          found = False 
          if char.lower() in alphabet:
               print "the index is: ",alphabet.index(char.lower()), " with shiftnum ",shiftnum
               index = (alphabet.index(char.lower())+ shiftnum)%26
               print "the NEW index is: ",index
               if char.isupper():
                    newtext.append(alphabet[index].upper())
               if char.islower():
                    newtext.append(alphabet[index].lower())
          else:
               newtext.append(char)
     return newtext

text = ''
for line in sys.stdin:
     text +=line

resultsArray =[]
for shiftnum in range(0,26):
     textArray = shifttext(text,shiftnum)
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






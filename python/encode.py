#!/usr/bin/python                                                               
import sys

alphabet = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]

input = sys.stdin.readline()
shiftnum =int(input.rstrip("\n"))
if shiftnum < 0:
     shiftnum = (shiftnum%26)
for line in sys.stdin:
     encrypted =[]
     line = line.rstrip("\n")
     line = list(line)
     for char in line:
          found = False 
          for letter in alphabet:
               if char.lower() == letter:
                    index = alphabet.index(char.lower())
                    rshift = (shiftnum+index)%26
                    if char.isupper():
                       encrypted.append(alphabet[rshift].upper())
                    if char.islower():
                       encrypted.append(alphabet[rshift].lower())
                    found = True
          if found == False:
               encrypted.append(char)
     print ''.join(encrypted)



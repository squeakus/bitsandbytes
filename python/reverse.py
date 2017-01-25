!#/usr/bin/python
import sys

for line in sys.stdin:
    line = line.rstrip("\n")
    revchars = list(line)
    revchars.reverse()              
    revchars = ''.join(revchars)
    print revchars

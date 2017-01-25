import sys
from math import cos, sin, radians

if len(sys.argv) != 4:
    print "rotatesimple.py x y angle"
    exit()

print sys.argv
x = float(sys.argv[1])
z = float(sys.argv[2])
deg = int(sys.argv[3])
newx = (x * cos(radians(deg))) - (z * sin(radians(deg)))
newz = (x * sin(radians(deg))) - (z * cos(radians(deg)))

print "x", newx, "y", newz

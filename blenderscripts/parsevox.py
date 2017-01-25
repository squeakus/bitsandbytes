"""Read the voxel data and get maxmin."""
import numpy as np
import sys

if len(sys.argv) < 2:
    print ("Usage %s <filename>" % sys.argv[0])
    sys.exit(1)


infile = open(sys.argv[1], 'r')
maxx, maxy, maxz, maxi = -np.inf, -np.inf, -np.inf, -np.inf
minx, miny, minz, mini = np.inf, np.inf, np.inf, np.inf

for line in infile:
    (x, y, z, i) = [float(x) for x in line.split(" ")[0:4]]

    if x > maxx:
        maxx = x
    if x < minx:
        minx = x
    if y > maxy:
        maxy = y
    if y < miny:
        miny = y
    if z > maxz:
        maxz = z
    if z < minz:
        minz = z
    if i > maxi:
        maxi = i
    if i < mini:
        mini = i

print 'max x:', maxx, 'y:', maxy, 'z:', maxz, 'i', maxi
print 'min x:', minx, 'y:', miny, 'z:', minz, 'i', mini

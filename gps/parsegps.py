# This program calculates the difference between 2 decimal GPS coords
# in metres. It is calibrated for 53 degrees north.

from itertools import combinations
from math import sqrt
from PIL import Image

gpsfile = open('out.txt','r')
lat, lon = None, None
imglocations = {}

#one degree equals 65904 metres
onedeg = 65904

for line in gpsfile:
    line = line.split()
    imgname = line[0]
    lat, lon = float(line[1]), float(line[2])
    imglocations[imgname] = [lat, lon]

tcount, ecount = 0,0 
    
for aloc, bloc in combinations(imglocations,2):
    alat, alon = imglocations[aloc][0], imglocations[aloc][1]
    blat, blon = imglocations[bloc][0], imglocations[bloc][1]
    dist = sqrt(((blat-alat)**2) + ((blon-alon)**2))
    metredist = onedeg * dist
 
    print "distance between",aloc,"and",bloc,"=",metredist
    tcount += 1
    if metredist > 70:
        ecount +=1
        img1 = Image.open('richviewnadir60/'+aloc)
        img2 = Image.open('richviewnadir60/'+bloc)
        img1 = img1.resize((400,300))
        img2 = img2.resize((400,300))
        img1.show()
        img2.show()
        raw_input("Press Enter to continue...")

print "%d images with no overlap out of %d" % (ecount, tcount)

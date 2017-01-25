import pylab

pointsfile = open('car_slices.csv','r')
ycoords = []
zcoords = []

for line in pointsfile:
    #only take a single slice
    if line.startswith('-132'):
        print "foundline"
        line = line.rstrip().split(',')
        x,y,z = float(line[0]),float(line[1]),float(line[2])
        ycoords.append(y)
        zcoords.append(z)
pylab.plot(ycoords,zcoords)
pylab.axis([-100,500,-100,500])
pylab.show()

# depth2cloud.py  - convert Kinect depth image into 3D point cloud, in PLY format
# Ben Bongalon (ben@borglabs.com)
#  @ TODO:
#    - change to GPL
#    - save in binary PLY format


import sys
from pylab import *

def main():
    if len(sys.argv) != 3:
        usage()
    imgfile = sys.argv[1]
    cloudfile = sys.argv[2]
    points, ranges = parse_image(imgfile)
    points = filter_points(points, ranges)
    create_ply(points, cloudfile)
    print 'Done!'

def usage():
    print '\ndepth2cloud  <img_file> <cloud_file>\n'
    exit(0)

def parse_image(imgfile):
    print 'Reading binary image...\n'
    scaleFactor = 0.0021
    minDistance = -10


    raw = fromfile(imgfile, 'H')
    # Compute depth (unit in cm) from raw 11-bit disparity value
    # According to ROS site
    depth = 100/(-0.00307*raw + 3.33)
    depth.resize(480,640)

    # Convert from pixel ref (i, j, z) to 3D space (x,y,z)
    points = []
    maxx, maxy, maxz = -1000, -1000, -1000
    minx, miny, minz = 1000, 1000, 1000


    for i in range(480):
        for j in range(640):
            x = depth[i][j]
            z = (i - 480 / 2) * (x + minDistance) * scaleFactor
            y = (640 / 2 - j) * (x + minDistance) * scaleFactor

            #filter crazy depths
            if not x > 150:
                if not x < 0:
                    points.append((x,y,z))
                    if x > maxx: maxx = x
                    if x < minx: minx = x
                    if y > maxy: maxy = y
                    if y < miny: miny = y
                    if z > maxz: maxz = z
                    if z < minz: minz = z

    ranges = {'maxx':maxx,'minx':minx, 'xrange':maxx-minx,
              'maxy':maxy,'miny':miny, 'yrange':maxy-miny,
              'maxz':maxz,'minz':minz, 'zrange':maxz-minz}
    return points, ranges

def filter_points(oldpoints, ranges):
    newpoints = []
    #limit the range
    maxx, maxy, maxz = -1000, -1000, -1000
    minx, miny, minz = 1000, 1000, 1000

    for oldpoint in oldpoints:
        x, y, z = oldpoint
        y = y - ranges['miny']        
        z = ranges['zrange'] - (z - ranges['minz'])
        
        if x > maxx: maxx = x
        if x < minx: minx = x
        if y > maxy: maxy = y
        if y < miny: miny = y
        if z > maxz: maxz = z
        if z < minz: minz = z

        newpoints.append((x,y,z))
    print "X", maxx,minx, (maxx - minx)/2
    print "Y", maxy,miny, (maxy - miny)/2
    print "Z",maxz, minz, (maxz - minz)/2
    return newpoints

def create_ply(points,filename):
    print 'Generating cloud file...\n'
    print "no of points", len(points)

    fc = open(filename, 'wt')
    fc.write('ply\n')
    fc.write('format ascii 1.0\n')
    fc.write('comment : created from Kinect depth image\n')
    fc.write('element vertex %d\n' % len(points))
    fc.write('property float x\n')
    fc.write('property float y\n')
    fc.write('property float z\n')
    fc.write('end_header\n')
    for point in points:
        x, y, z = point
        fc.write("%f  %f  %f\n" % (x, y, z))
    fc.close()


if __name__=='__main__':
    main()

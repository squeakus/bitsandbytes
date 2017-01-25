import sys, time, subprocess
from pylab import *


def main():
  orientation = "moo"
  points1 = parse_image(orientation+'_001.bin')
  points2 = parse_image(orientation+'_002.bin')
#  points3 = parse_image(orientation+'_003.bin')

  allpoints = points1
  cleanpoints = []
  print "points len", len(allpoints)
  doubles = 0
  for point in points2:
    if point in allpoints:
      doubles += 1
    else:
      cleanpoints.append(point)
    if doubles % 10000 == 0:
        print doubles, len(cleanpoints)
  print "number of redundant points2", doubles

#  doubles = 0
#  for point in points3:
#    if point in allpoints:
#      doubles += 1
#    else:
#      allpoints.append(point)
#  print "number of redundant points3", doubles
  print "allpoints", len(allpoints)
    
def grabpoints(orientation):
  run_cmd('glgrab '+orientation)
  points1 = parse_image(orientation+'_001.bin')
  points2 = parse_image(orientation+'_002.bin')
  points3 = parse_image(orientation+'_003.bin')
  print "pointsize", len(points1), len(points2), len(points3)
  allpoints = points1 + points2 + points3
  print "allpoints", len(allpoints)
  allpoints = filter_points(allpoints)
  create_ply(allpoints, orientation+'.ply')


def parse_image(imgfile):
  print 'Reading binary image', imgfile
  minDistance = -10
  scaleFactor = 0.0021

  raw = fromfile(imgfile, 'H')
  # Compute depth (unit in cm) from raw 11-bit disparity value
  # According to ROS site
  depth = 100/(-0.00307*raw + 3.33)
  depth.resize(480,640)
  
  # Convert from pixel ref (i, j, z) to 3D space (x,y,z)
  points = []
  for i in range(480):
    for j in range(640):
      z = depth[i][j]
      x = (i - 480 / 2) * (z + minDistance) * scaleFactor
      y = (640 / 2 - j) * (z + minDistance) * scaleFactor
      points.append((x,y,z))
  return points

def filter_points(oldpoints):
  newpoints = []
  for oldpoint in oldpoints:
    x, y, z = oldpoint
    if not z > 300:
      if not z < 200:
        newpoints.append((x,y,z))
  return newpoints

def run_cmd(cmd, debug = False):
    """execute commandline command cleanly"""
    if debug:
        print cmd
    else:
        cmd += " > /dev/null 2>&1"
    process = subprocess.Popen(cmd, shell=True,
                               stdout=subprocess.PIPE,
                               stdin=subprocess.PIPE)
    result = process.communicate()
    return result
  
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

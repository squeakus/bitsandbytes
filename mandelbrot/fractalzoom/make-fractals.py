#!/usr/bin/python2
# (C) 2008 Sean Brennan

# Drives the c++ fractal generator by providing batching (looping) and
# detection of fractals alread rendered. Also provides logic for the zoom.
# The bounding box is shrunk by raising the zoom value to an exponential that
# is equal to the frame number.
#
# Args are:
#   -s: start frame
#   -e: end frame
#   -a: anti alias count

import os
import sys

start_f = 1 # start frame
end_f = 200 # end frame
xres = 900 # output x resolution
yres = 900 # output y resolution
anti = 1 # number of anti-alias loops

if len(sys.argv) > 1:
  args = sys.argv
  for i in range(len(args)):
    if args[i] == '-s':
      i += 1
      start_f = int(args[i])
    if args[i] == '-e':
      i += 1
      end_f = int(args[i])
    if args[i] == '-a':
      i += 1
      anti = int(args[i])

print 'start frame: %d end frame: %d res: %d %d ' % (
  start_f, end_f, xres,  yres)

# Hard coded zoom point:
nice_point = [ -.743643887037151, .131825904205330]
radius = 0.01

shrink_factor = 0.98 # how fast zoom contracts.
if start_f > 1:
  radius *= shrink_factor ** (start_f - 1)

ex = "./make-them-fractals"

def not_cached(outdir, fname):
  if os.system('ls %s' % (fname)):
    return True
  else:
    return False

# generates bounding box tuple for point and radius
def wince(point, rad):
  x_ = point[0] - rad
  y_ = point[1] - rad
  x = point[0] + rad
  y = point[1] + rad

  return (x_, y_, x, y)

outdir = 'flat/'
if os.system('ls %s' % outdir):
  os.system('mkdir %s' % outdir)

for i in xrange(start_f, end_f + 1):
  re_, im_, re, im = wince(nice_point, radius)
  outname = "%s/frame_%.4d-a%d.ppm" % (outdir, i, anti)
  #print " -0.24567 0.64196 -0.24679 0.64203 600 600 f.ppm"
  if not_cached(outdir, outname):
    cmd =  "%s %.16f %.16f %.16f %.16f %d %d %d %s" % (
     ex,  re_, im_, re, im, xres, yres, anti, outname)
    print cmd
    os.system(cmd)
  else:
    print 'Found file: %s, skipping...' % outname
  radius *= shrink_factor

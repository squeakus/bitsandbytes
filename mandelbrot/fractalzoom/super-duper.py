#!/usr/bin/python
"(C) 2008 Sean Brennan.  Main program to drive blender"

import os
import time
import sys

pid = os.getpid()
start_frame = 1
end_frame = 2

xysize = 900

anti_alias_level = 4

batch = True
if len(sys.argv) > 1:
  argv = sys.argv
  print "flags:"
  i = 0
  while i < len(argv):
    a = argv[i]
    print a
    if a == '-i':
      batch = False
      end_frame = start_frame
    if a == '-f':
      start_frame = int(argv[i + 1])
      end_frame = start_frame
      i += 1
    if a == '-s':
      start_frame = int(argv[i + 1])
      i += 1
    if a == '-e':
      end_frame = int(argv[i + 1])
      i += 1
    if a == '-z':
      xysize = int(argv[i + 1])
      i += 1
    i += 1
else:
  pass


num_frames = 2000

start_x = 4200
start_y = 5700

end_x = 8000
end_y = 7000

width = 800
height = 800


total_x = float(end_x - start_x)
total_y = float(end_y - start_y)


step_x = total_x / float(num_frames)
step_y = total_y / float(num_frames)

def run_command(cmd):
  print('running %s' % cmd)
  os.system(cmd)

xh = int(start_x + step_x * start_frame)
yh = int(start_y + step_y * start_frame)

pyout = '/tmp/p-%d' % pid
cmd = "sed -e 's/XXXXXXXX/%d/' template.py > %s" % (pid, pyout)
run_command(cmd)

cmd = "chmod a+x %s" % pyout
run_command(cmd)

cwd = os.getcwd()
for i in xrange(start_frame, end_frame + 1):
  start = time.time()
  # remove old texture and heightfield images
  cmd = '/bin/rm -f /tmp/t-%d.tga /tmp/r-%d.tga' % (pid, pid)
  run_command(cmd)
  cmd = 'make-fractals.py -s %d -e %d -a %d' % (i, i, anti_alias_level) 
  run_command(cmd)
  
  source_ppm = '%s/flat/frame_%.4d-a%d.ppm' % (cwd, i, anti_alias_level)
  # create heightfield image 
  cmd = 'cat %s | pnmscale -xysize %d %d' % (source_ppm, xysize, xysize)
  #cmd += '|  pnmsmooth -width 3 -height 3'
  #cmd += '|  pnmsmooth -width 3 -height 3'
  cmd +=  '|  ppmtotga > /tmp/r-%d.tga' % pid
  #cmd = 'ln -s %s/tga/height_%0.4d.tga /tmp/r-%d.tga' % (cwd, i, pid)
  run_command(cmd)
  # create texture
  cmd = 'pnmarith -add %s %s/tex/purple9.ppm |ppmtotga > /tmp/t-%d.tga' % (
         source_ppm, cwd, pid)
  run_command(cmd)
  #cmd = '/bin/rm -f /tmp/tp-%d.ppm' %  pid
  #run_command(cmd)
  if batch:
    cmd = 'blender blank.blend -o //render_# -P %s -f %d -b' % (
         pyout, i)
  else:
    cmd = 'blender blank.blend  -P %s ' %  pyout
  run_command(cmd)
  print 'Time to Execute: %3.2f seconds' % (start - time.time())
cmd = '/bin/rm -f /tmp/r-%d.tga %s /tmp/t-%d.pgm' % ( pid, pyout, pid)
run_command(cmd)

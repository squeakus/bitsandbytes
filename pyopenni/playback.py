#!/usr/bin/python
## The equivalent of:
##  "Recording and Playing Data (Playing)"
## in the OpenNI user guide.

"""
The following code opens up the recording file created by record.py, 
and takes the depth generator that was created for this purpose. 

It will replay the results of the recorded data, in a loop.
"""


from openni import *

ctx = Context()
ctx.init()

# Open recording 
ctx.open_file_recording("tempRec.oni")

ctx.start_generating_all()
depth = ctx.find_existing_node(NODE_TYPE_DEPTH)


while True:
    # Update to next frame
    nRetVal = ctx.wait_one_update_all(depth)

    depthMap = depth.map

    # Get the coordinates of the middle pixel
    x = depthMap.width / 2
    y = depthMap.height / 2
    
    # Get the pixel at these coordinates
    pixel = depthMap[x,y]

    print "The middle pixel is %d millimeters away." % pixel



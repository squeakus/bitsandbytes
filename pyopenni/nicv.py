from openni import *
import numpy as np
import cv2
 
# Initialise OpenNI
context = Context()
context.init()

# Create a depth generator to access the depth stream
depth = DepthGenerator()
depth.create(context)
depth.set_resolution_preset(RES_VGA)
depth.fps = 30

# Start Kinect
context.start_generating_all()

print "now reading from camera, press q to quit"

while True:
    #get latest data frame
    context.wait_any_update_all()
    #depth.wait_and_update_data()

    # Create array from the raw depth map string
    frame = np.fromstring(depth.get_raw_depth_map_8(), "uint8").reshape(480, 640)
    oldframe = frame
    # Render in OpenCV
    cv2.imshow("image", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

import freenect
import cv2
import numpy as np
 
"""
Grabs a depth map from the Kinect sensor and creates an image from it.
"""
def getDepthMap():    
    depth, timestamp = freenect.sync_get_depth()
 
    np.clip(depth, 0, 2**10 - 1, depth)
    depth >>= 2
    depth = depth.astype(np.uint8)
 
    return depth
 
while True:
    depth = getDepthMap()
 
    blur = cv2.GaussianBlur(depth, (5, 5), 0)
 
    cv2.imshow('image', blur)
    cv2.waitKey(10)

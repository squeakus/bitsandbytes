import cv2
import numpy as np
 
img = cv2.imread('coins.png')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
#kp = sift.detect(gray,None)
kp, desc = sift.detectAndCompute(gray,None)
cv2.drawKeypoints(gray,kp, img)

cv2.imwrite('sift_keypoints.jpg',img)
for idx,keypoint in enumerate(kp):
    print idx, keypoint.pt, keypoint.angle, keypoint.size
    print "shape", desc.shape
    print desc[idx]
    print ""

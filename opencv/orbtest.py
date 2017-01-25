import cv2
import numpy as np

img = cv2.imread('coins.png')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create()
kp, desc = orb.detectAndCompute(gray,None)

cv2.drawKeypoints(gray,kp, img)
cv2.imwrite('orb_keypoints.jpg',img)

if len(desc[0]) < 128:
    print "descriptor length is too short:", len(desc[0])
    newdesc = []
    for idx, ds in enumerate(desc):
        newarray = desc[idx].copy()
        newarray.resize(128)
        newdesc.append(newarray)
    desc =newdesc

for idx,keypoint in enumerate(kp):
    print idx, keypoint.pt, keypoint.angle, keypoint.size
    print "shape", desc[idx].shape
    print desc[idx]


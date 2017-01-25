import cv2, sys
import numpy as np

imagename = sys.argv[1]
img = cv2.imread(imagename)
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

orb = cv2.ORB()
kp = orb.detect(gray,None)
img=cv2.drawKeypoints(gray,kp)
cv2.imwrite('orb_'+imagename,img)

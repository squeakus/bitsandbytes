import cv2
import numpy as np

c = cv2.VideoCapture('a.avi')
_,f = c.read()

avg1 = np.float32(f)
avg2 = np.float32(f)

# loop over images and estimate background 
for x in range(0,4):
    _,f = c.read()

    cv2.accumulateWeighted(f,avg1,1)
    cv2.accumulateWeighted(f,avg2,0.01)

    res1 = cv2.convertScaleAbs(avg1)
    res2 = cv2.convertScaleAbs(avg2)

    cv2.imshow('img',f)
    cv2.imshow('avg1',res1)
    cv2.imshow('avg2',res2)
    k = cv2.waitKey(0) & 0xff
    if k == 5:
        break

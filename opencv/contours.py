import numpy as np
import cv2

im = cv2.imread('star.jpg')
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

print "contours", len(contours), "boundary points", len(contours[0])

#line 3 pixels wide
cv2.imshow('image',im)
k = cv2.waitKey(0) & 0xFF

cv2.drawContours(im,contours,-1,(0,255,0),3)

cv2.imshow('image',im)
k = cv2.waitKey(0) & 0xFF

#-1 is used to fill the object
cv2.drawContours(im,contours,-1,(0,0,255),-1)

cv2.imshow('image',im)
k = cv2.waitKey(0) & 0xFF

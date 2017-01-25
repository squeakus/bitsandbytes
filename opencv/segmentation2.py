#http://stackoverflow.com/questions/11294859/how-to-define-the-markers-for-watershed-in-opencv
import cv2
import numpy as np

img = cv2.imread('hardseg.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

fg = cv2.erode(thresh,None,iterations = 2)
bgt = cv2.dilate(thresh,None,iterations = 3)
ret,bg = cv2.threshold(bgt,1,128,1)

marker = cv2.add(fg,bg)

cv2.imshow('marked stuff',marker)
k = cv2.waitKey(0) & 0xFF


marker32 = np.int32(marker)

cv2.imshow('32 bit',marker)
k = cv2.waitKey(0) & 0xFF

cv2.watershed(img,marker32)
m = cv2.convertScaleAbs(marker32)

cv2.imshow('boundaries',m)
k = cv2.waitKey(0) & 0xFF


ret,thresh = cv2.threshold(m,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
res = cv2.bitwise_and(img,img,mask = thresh)


cv2.imshow('boundaries',res)
k = cv2.waitKey(0) & 0xFF

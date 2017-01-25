import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('coins.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

# Marker labelling
#ret, markers = cv2.connectedComponents(sure_fg)
ret, markers = cv2.findContours(sure_fg, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

# Now, mark the region of unknown with zero
markers[unknown==255] = 0

#cv2.imshow('thresh',thresh)
#cv2.imshow('opening',opening)
#cv2.imshow('sure_bg',sure_bg)
#cv2.imshow('sure_fg',sure_fg)
#cv2.imshow('unknown',unknown)
cv2.imshow('markers',markers)

k = cv2.waitKey(0) & 0xFF

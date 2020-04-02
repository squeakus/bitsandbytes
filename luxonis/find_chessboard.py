#!/usr/bin/python
# coding=utf-8

# Base Python File (find_chessboard.py)
# Created: Wed 01 Apr 2020 13:36:37 IST
# Version: 1.0
#
# This Python script was developped by Jonathan Byrne.
# You are free to copy, adapt or modify it.
#
# (ɔ) Jonathan Byrne <jonathanbyrn@gmail.com>

import numpy as np
import cv2
import glob
import time

# Chessboard detection termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
chessboard_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
        
    ret, img = cap.read()

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (0, 0), fx = 0.3, fy = 0.3) 

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(small, (9,6),chessboard_flags)

    # If found, add object points, image points (after refining them)
    if ret == True:
    # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),chessboard_flags)
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)
    end = time.time()
    fps = round(1 / (end - start), 1)
    # cv2.putText(img, "FPS:"+str(fps), (10, 30),
    #     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)  
    cv2.imshow('img',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()


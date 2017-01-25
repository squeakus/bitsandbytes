#!/usr/bin/env python

'''
camera calibration for distorted images with chess board samples
reads distorted images, calculates the calibration and write undistorted images

usage:
    calibrate.py [--debug <output path>] [--square_size] [<image mask>]

default values:
    --debug:    ./output/
    --square_size: 1.0
    <image mask> defaults to ../data/left*.jpg
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2

# local modules
from common import splitfn

# built-in modules
import os

if __name__ == '__main__':
    import sys
    import getopt
    from glob import glob

    img_names = glob("/home/jonathan/data/scans/railway/*.JPG")
    #rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)
    rms =  2.5830626755
    camera_matrix = np.array([[2.31549190e+03, 0.00000000e+00, 2.04127254e+03],
                     [0.00000000e+00, 2.34319145e+03, 1.46246521e+03],
                     [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

    dist_coefs = np.array([0.01127417, -0.03455549,
                           -0.00543075, 0.00240345, 0.03604999])
    print("\nRMS:", rms)
    print("camera matrix:\n", camera_matrix)
    print("distortion coefficients: ", dist_coefs.ravel())

    # undistort the image with the calibration
    print('')
    for img_found in img_names:
        img = cv2.imread(img_found)

        h,  w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))

        dst = cv2.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)

        # crop and save the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        outfile = img_found.rstrip('.JPG') + '_undistorted.jpg'
        print('Undistorted image written to: %s' % outfile)
        cv2.imwrite(outfile, dst)

    cv2.destroyAllWindows()

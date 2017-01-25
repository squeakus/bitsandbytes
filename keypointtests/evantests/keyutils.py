# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2, os

def init_feature(name):
    '''
    WHAT ABOUT BRIEF!
    latch, daisy, freak do not have detect and compute implemented
    '''
    detector = None
    chunks = name.split('-')
    if chunks[0] == 'sift':
        detector = cv2.xfeatures2d.SIFT_create()
        norm = cv2.NORM_L2
    elif chunks[0] == 'surf':
        detector = cv2.xfeatures2d.SURF_create()
        norm = cv2.NORM_L2
    elif chunks[0] == 'orb':
        detector = cv2.ORB_create(100000)
        norm = cv2.NORM_HAMMING
    elif chunks[0] == 'akaze':
        detector = cv2.AKAZE_create()
        norm = cv2.NORM_HAMMING
    elif chunks[0] == 'brisk':
        detector = cv2.BRISK_create()
        norm = cv2.NORM_HAMMING

    if detector == None:
        print("couldn't find detector:", name)
        exit()

    if 'flann' in chunks:
        if norm == cv2.NORM_L2:
	    FLANN_INDEX_KDTREE = 0
            flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        else:
	    FLANN_INDEX_LSH = 6
            flann_params= dict(algorithm = FLANN_INDEX_LSH,
                               table_number = 6, # 12
                               key_size = 12,     # 20
                               multi_probe_level = 1) #2
        matcher = cv2.FlannBasedMatcher(flann_params, {})
    else:
        matcher = cv2.BFMatcher(norm)
    print("Processing images using:", chunks[0])
    return detector, matcher

def find_images(direct):
    '''
    Checks for image extensions in a folder
    returns filename and filename + directory tuples
    '''
    extensions =["ppm","PPM",
            "pbm","PBM",
            "pgm","PGM",
            "png","PNG",
            "jpg","JPG",#jpeg group extensions
            "jpeg","JPEG",
            "jpe","JPE",
            "tiff","TIFF",
            "tif","TIF",
            "bmp","BMP",
            "sr","SR",#Sun raster format
            "ras","RAS",
            "jp2","JP2",#Jasper images
    ]

    files = os.listdir(direct)
    if direct[-1] == os.sep:
        directory = direct
    else:
        directory = direct+os.sep

    image_files = []
    for filen in files:
        full_file =  directory+filen
        filename,ext = os.path.splitext(os.path.basename(filen))
        ext=ext.replace(".","")
        if ext in extensions:
            image_files.append([filename,full_file])

    return image_files

def filter_matches(kp1, kp2, matches, ratio = 0.8):
    '''
    Filters matches according to David Lowe's paper, if the second closest
    match is less distance than the given ratio, keep it 
    '''
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append( kp1[m.queryIdx] )
            mkp2.append( kp2[m.trainIdx] )
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)
    return p1, p2, kp_pairs

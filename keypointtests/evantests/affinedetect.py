#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import cv2, sys
import itertools as it
from multiprocessing.pool import ThreadPool
from keyutils import init_feature, filter_matches

def affine_skew(tilt, phi, img, mask=None):
    '''
    affine_skew(tilt, phi, img, mask=None) -> skew_img, skew_mask, Ai
    Ai - is an affine transform matrix from skew_img to img
    '''
    h, w = img.shape[:2]
    if mask is None:
        mask = np.zeros((h, w), np.uint8)
        mask[:] = 255
    A = np.float32([[1, 0, 0], [0, 1, 0]])
    if phi != 0.0:
        phi = np.deg2rad(phi)
        s, c = np.sin(phi), np.cos(phi)
        A = np.float32([[c,-s], [ s, c]])
        corners = [[0, 0], [w, 0], [w, h], [0, h]]
        tcorners = np.int32( np.dot(corners, A.T) )
        x, y, w, h = cv2.boundingRect(tcorners.reshape(1,-1,2))
        A = np.hstack([A, [[-x], [-y]]])
        img = cv2.warpAffine(img, A, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if tilt != 1.0:
        s = 0.8*np.sqrt(tilt*tilt-1)
        img = cv2.GaussianBlur(img, (0, 0), sigmaX=s, sigmaY=0.01)
        img = cv2.resize(img, (0, 0), fx=1.0/tilt, fy=1.0, interpolation=cv2.INTER_NEAREST)
        A[0] /= tilt
    if phi != 0.0 or tilt != 1.0:
        h, w = img.shape[:2]
        mask = cv2.warpAffine(mask, A, (w, h), flags=cv2.INTER_NEAREST)
    Ai = cv2.invertAffineTransform(A)
    return img, mask, Ai


def affine_detect(detector, img, mask=None, pool=None):
    '''
    affine_detect(detector, img, mask=None, pool=None) -> keypoints, descrs

    Apply a set of affine transormations to the image, detect keypoints and
    reproject them into initial image coordinates.
    See http://www.ipol.im/pub/algo/my_affine_sift/ for the details.

    ThreadPool object may be passed to speedup the computation.
    '''
    params = [(1.0, 0.0)]
    for t in 2**(0.5*np.arange(1,6)):
        for phi in np.arange(0, 180, 72.0 / t):
            params.append((t, phi))

    def f(p):
        t, phi = p
        timg, tmask, Ai = affine_skew(t, phi, img)
        keypoints, descrs = detector.detectAndCompute(timg, tmask)
        for kp in keypoints:
            x, y = kp.pt
            kp.pt = tuple( np.dot(Ai, (x, y, 1)) )
        if descrs is None:
            descrs = []
        return keypoints, descrs

    keypoints, descrs = [], []
    if pool is None:
        print("Not using threads")
        ires = it.imap(f, params)
    else:
        ires = pool.imap(f, params)

    for i, (k, d) in enumerate(ires):
        print('affine sampling: %d / %d\r' % (i+1, len(params)), end='')
        keypoints.extend(k)
        descrs.extend(d)

    return keypoints, np.array(descrs)

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print ("Usage %s <image1> <image2> <keypoint_detector>" % sys.argv[0])
        exit()

    img1 = cv2.imread(sys.argv[1], 0) # queryImage
    img2 = cv2.imread(sys.argv[2], 0) # trainImage
    detectstr = sys.argv[3]
    detector, matcher = init_feature(detectstr)
    print('using', detectstr)
    cpucount = cv2.getNumberOfCPUs() / 2
    pool=ThreadPool(processes = cpucount)
    print("No. of CPUs: %d" % cpucount)
    kp1, desc1 = affine_detect(detector, img1, pool=None)
    kp2, desc2 = affine_detect(detector, img2, pool=None)
    print('img1 - %d features, img2 - %d features' % (len(kp1), len(kp2)))

    raw_matches = matcher.knnMatch(desc1, trainDescriptors = desc2, k = 2) #2
    p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches)

    if len(p1) >= 4:
        H, matchMask = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
        print('%d / %d inliers/matched' % (np.sum(matchMask), len(matchMask)))
        # do not draw outliers (there will be a lot of them)
        kp_pairs = [kpp for kpp, flag in zip(kp_pairs, matchMask) if flag]
    else:
        H, matchMask = None, None
        print('%d matches found, not enough for homography estimation' % len(p1))

    # cv2.drawMatchesKnn expects list of lists as matches.
    all_match = []
    for m,n in raw_matches:
        all_match.append([m])
 
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2, all_match, None, flags=2)
    cv2.imwrite('match'+detectstr+'.png',img3)
    cv2.imshow('asift result',img3)

    cv2.waitKey()
    cv2.destroyAllWindows()

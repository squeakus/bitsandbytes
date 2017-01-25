import numpy as np
import cv2, sys
from matplotlib import pyplot as plt

def main():
    img1 = cv2.imread(sys.argv[1], 0) # queryImage
    img2 = cv2.imread(sys.argv[2], 0) # trainImage
    detectstr = sys.argv[3]
    detector = create_detector(detectstr)

    # find the keypoints and descriptors with DETECTOR
    kp1, des1 = detector.detectAndCompute(img1,None)
    kp2, des2 = detector.detectAndCompute(img2,None)
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    # Apply ratio test
    good = []
    good2 = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            #print [m], m
            good.append([m])
            good2.append(m)

    # cv2.drawMatchesKnn expects list of lists as matches.
#    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good, None, flags=2)
#    cv2.imwrite('match'+detectstr+'.png',img3)
#    cv2.imshow('result',img3)
#    cv2.waitKey()
#    cv2.destroyAllWindows()


    M, matchesMask = validate(kp1,kp2,good2,img1, img2)

    draw_params = dict(matchColor = (0,255,0), # draw matches in green 
                       singlePointColor = None,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)
    
    #attempt to draw inliers on top of matches
    img4 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good2, None, **draw_params)
    cv2.imshow('inliers',img4)
    cv2.waitKey()
    cv2.destroyAllWindows()



    print "img1 keypoints", len(kp1), "img2 keypoints", len(kp2)
    print "matches", len(good)

    
def validate(kp1,kp2,good, img1, img2):
   src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
   dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
   M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
   matchesMask = mask.ravel().tolist()

   h,w = img1.shape
   pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
   dst = cv2.perspectiveTransform(pts,M)
   
   img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
   
   print "inliers:", matchesMask.count(1)
   return M, matchesMask

def create_detector(detect):
    detector = None
    #free ones first
    if detect == 'orb':
        detector = cv2.ORB_create(nfeatures=100000)
    if detect == 'akaze':
        detector = cv2.AKAZE_create()
    if detect == 'brisk':
        detector = cv2.BRISK_create()

    #proprietary second
    if detect == 'sift':
        detector = cv2.xfeatures2d.SIFT_create()
    if detect == 'surf':
        detector = cv2.xfeatures2d.SURF_create()
    if detect == 'freak':
        detector = cv2.xfeatures2d.FREAK_create()
    if detect == 'latch':
        detector = cv2.xfeatures2d.LATCH_create()
    if detect == 'daisy':
        detector = cv2.xfeatures2d.DAISY_create()
    if detect == 'lucid':
        detector = cv2.xfeatures2d.LUCID_create()

    if detector == None:
        print "Could not find a detector of that name"
        exit()

    print "Processing images using:", detect
    return detector


if __name__=='__main__':
    if len(sys.argv) < 4:
        print ("Usage %s <image1> <image2> <keypoint_detector>" % sys.argv[0])
        sys.exit(1)
    main()

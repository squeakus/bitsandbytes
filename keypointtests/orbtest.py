import cv2
import sys
import numpy as np

def main(pointcnt):
    img = cv2.imread('home.jpg')

    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    outfile = "orb-"+pointcnt+'.jpg'

    detector = cv2.ORB_create(int(pointcnt))
    #detector = cv2.xfeatures2d.SIFT_create()
    #kp = sift.detect(gray,None)
    kp = detector.detect(gray,None)
    print "no. of keypoints", len(kp)
    #img=cv2.drawKeypoints(gray,kp)
    cv2.drawKeypoints(gray,kp, img)

    cv2.imwrite(outfile,img)

if __name__=='__main__':
    if len(sys.argv) < 2:
        print ("Usage %s <pointcnt>" % sys.argv[0])
        exit()
    main(sys.argv[1])

import cv2, sys
import numpy as np
from matplotlib import pyplot as plt

def main():
    if len(sys.argv) < 2:
        print "usage: python golfdetect.py <imagename>"
        exit()
    original = cv2.imread(sys.argv[1],0)
    edges = cv2.Canny(original,100,200)

    #plt.subplot(121),plt.imshow(img,cmap = 'gray')
    #plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    #plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    #plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()
    cv2.imshow('edge detect',edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #imgray = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(edges,127,255,0)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for idx, contour in enumerate(contours):
        if len(contour) > 20:
            area = cv2.contourArea(contour)
            if area > 20:
                isconvex = cv2.isContourConvex(contour)
                circle = cv2.minEnclosingCircle(contour)
                cv2.drawContours(original,contours,idx,(0,255,0),3)
                print "contour", idx, "length", len(contour), "area", area, "convex", isconvex, "circle", circle


    cv2.imshow('image',original)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()

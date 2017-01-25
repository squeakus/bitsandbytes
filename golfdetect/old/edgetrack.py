import cv2, sys
import numpy as np

def main():
    if len(sys.argv) < 2:
        print "usage: python golfdetect.py <imagename>"
        exit()
    original = cv2.imread(sys.argv[1],1)
    img = cv2.imread(sys.argv[1],0)
    edges = cv2.Canny(img,100,200)

    # Create a black image, a window
    cv2.namedWindow('Edges')

    # create trackbars for color change
    cv2.createTrackbar('Min','Edges',0,1000,nothing)
    cv2.createTrackbar('Max','Edges',0,1000,nothing)

    while(1):
        cv2.imshow('Edges',edges)
        k = cv2.waitKey(1) & 0xFF
        #print k
        if k == 27:
            break
        if k == 32 :
            process_edges(edges, original, img)

        # get current positions of four trackbars
        edgemin = cv2.getTrackbarPos('Min','Edges')
        edgemax = cv2.getTrackbarPos('Max','Edges')
        edges = cv2.Canny(img,edgemin,edgemax)

    cv2.destroyAllWindows()


def process_edges(edges, original, img):
    cv2.namedWindow('Contours')
    lengthlimit = 20
    arealimit = 20 

    # create trackbars for color change
    cv2.createTrackbar('length','Contours',0,200,nothing)
    cv2.createTrackbar('area','Contours',0,2000,nothing)
 
    ret,thresh = cv2.threshold(edges,127,255,0)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    while(1):
        lengthlimit = cv2.getTrackbarPos('length','Contours')
        arealimit = cv2.getTrackbarPos('area','Contours')

        for idx, contour in enumerate(contours):
            if len(contour) > lengthlimit:
                area = cv2.contourArea(contour)
                if area > arealimit:
                    isconvex = cv2.isContourConvex(contour)
                    circle = cv2.minEnclosingCircle(contour)
                    cv2.drawContours(original,contours,idx,(0,255,0),2)
                    cv2.drawContours(img,contours,idx,(0,255,0),2)
                    print "contour", idx, "length", len(contour), "area", area, "convex", isconvex, "circle", circle
            else:
                cv2.drawContours(original,contours,idx,(0,0,255),2)

        k = cv2.waitKey(1) & 0xFF
        #print k
        if k == 27:
            break

        cv2.imshow('Contours',original)

    cv2.imshow('Final Contours',img)
    cv2.waitKey(0)
    return


def nothing(x):
    pass


if __name__=='__main__':
    main()

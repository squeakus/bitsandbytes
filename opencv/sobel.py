import cv2, sys
import numpy as np

def main():
    if len(sys.argv) < 2:
        print("usage: python edgedetector.py <imagename>")
        exit()
    # read in the image as color and greyscale
    color = cv2.imread(sys.argv[1],1)
    # remove noise
    color = cv2.GaussianBlur(color,(3,3),0)
    cv2.imwrite("contours0.png", color)
    color = find_contours(color, (255, 255, 255), True)
    cv2.imwrite("contours1.png", color)
    color = find_contours(color, (255, 255, 255))
    cv2.imwrite("contours2.png", color)
    color = find_contours(color, (0, 255, 0))
    cv2.imwrite("contours3.png", color)

def find_edges(img):    
    # convolute with proper kernels
    sobel = cv2.Sobel(img,cv2.CV_64F,1,1,ksize=13)  # x
    cv2.imwrite("laplace.png", laplacian)
    cv2.imwrite("sobel.png", sobel)
    cv2.imwrite("sobely.png", sobely)


#find_edges(img, colorimg)
def find_contours(img, color, infill=False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,gray = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print("found contours:", len(contours))

    if infill:
        print("INFILL")
        cv2.drawContours(img, contours, -1, color, -1)
    else:
        cv2.drawContours(img, contours, -1, color, 1)
    return img


if __name__=='__main__':
    main()

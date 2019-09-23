import cv2, sys
import numpy as np

def main():
    if len(sys.argv) < 2:
        print("usage: python edgedetector.py <imagename>")
        exit()
    # read in the image as color and greyscale
    color = cv2.imread(sys.argv[1],1)
    gray = cv2.imread(sys.argv[1],0)

    # remove noise
    img = cv2.GaussianBlur(gray,(3,3),0)

    # convolute with proper kernels
    laplacian = cv2.Laplacian(img,cv2.CV_64F)
    sobel = cv2.Sobel(img,cv2.CV_64F,1,1,ksize=13)  # x
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)  # y
    cv2.imwrite("laplace.png", laplacian)
    cv2.imwrite("sobel.png", sobel)
    cv2.imwrite("sobely.png", sobely)


    #find_edges(img, colorimg)

if __name__=='__main__':
    main()

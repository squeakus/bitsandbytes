import cv2
import numpy as np


def main():
    img = cv2.imread("blob.png", cv2.IMREAD_GRAYSCALE)
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 1
    params.maxArea = 1000
    params.minDistBetweenBlobs = 1
    params.filterByCircularity = False
    params.filterByColor = False
    # params.blobColor = 255
    params.filterByConvexity = False
    params.filterByInertia = False
    detector = cv2.SimpleBlobDetector_create(params)

    keypoints = detector.detect(img)

    blank = np.zeros((1, 1))
    blobs = cv2.drawKeypoints(img, keypoints, blank, (0, 255, 255), cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    cv2.imshow("Blobs", blobs)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()

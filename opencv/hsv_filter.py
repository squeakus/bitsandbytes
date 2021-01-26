import cv2
import numpy as np


def main():
    image = cv2.imread("median.jpg")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 0, 0], np.uint8)
    upper = np.array([4, 255, 255], np.uint8)

    mask = cv2.inRange(hsv, lower, upper)
    cv2.imshow("threshold", mask)
    cv2.waitKey(0)
    cv2.imwrite("mask.jpg", mask)


if __name__ == "__main__":
    main()
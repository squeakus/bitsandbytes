import cv2
import numpy as np


def main():
    image = cv2.imread("median.jpg")[700:, :]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 0, 0], np.uint8)
    upper = np.array([4, 255, 255], np.uint8)

    mask = cv2.inRange(hsv, lower, upper)
    cv2.imwrite("before.jpg", image)
    cv2.imwrite("mask.jpg", mask)


if __name__ == "__main__":
    main()

import cv2
import sys


def main(imagename):
    image = cv2.imread(imagename)
    grey = cv2.imread(imagename, 0)
    ret, thresh = cv2.threshold(grey, 10, 255, 0)
    threshname = imagename.replace(".jpg", "_thresh.jpg")
    cv2.imwrite(threshname, thresh)

    # calc those contours!
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # only compute for maximum contour
    maxarea, maxcnt = 0, 0

    # for contour in contours:
    #     area = cv2.contourArea(contour)
    #     if area > maxarea:
    #         maxarea = area
    #         maxcnt = contour

    # image = cv2.drawContours(image, contours, -1, (0, 255, 0), -1)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for contour in contours:
        area = cv2.contourArea(contour)
        print(area)

    M = cv2.moments(contours[0])
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    hull = cv2.convexHull(contours[0], False)

    epsilon = 0.001 * cv2.arcLength(contours[0], True)
    approximate = cv2.approxPolyDP(contours[0], epsilon, True)
    print(approximate)
    image = cv2.drawContours(image, [approximate], 0, (0, 0, 255), 2)
    image = cv2.circle(image, (cX, cY), 7, (0, 255, 0), -1)

    outname = imagename.replace(".jpg", "_out.jpg")
    cv2.imwrite(outname, image)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print("pass in image name")

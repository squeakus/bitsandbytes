import numpy as np
import cv2


def main():
    im = cv2.imread("bumpclean.png")
    dim = (im.shape[1] * 4, im.shape[0] * 4)
    im = cv2.resize(im, dim, interpolation=cv2.INTER_AREA)
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # print("contours", len(contours), "boundary points", len(contours[0]))

    # line 3 pixels wide
    cv2.imshow("threshold", thresh)
    cv2.waitKey(0)

    cv2.drawContours(im, contours, -1, (0, 0, 255), -1)

    # font = cv2.FONT_HERSHEY_SIMPLEX
    font = cv2.FONT_HERSHEY_PLAIN
    for cnt in contours:
        M = cv2.moments(cnt)
        # print(M)
        area = M["m00"]
        if area < 5:
            continue
        print(M["m10"], M["m00"])
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        text = str(cx) + " " + str(cy)
        cv2.putText(im, text, (cx + 10, cy + 10), font, 1, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow("threshold", im)
    cv2.waitKey(0)

    # cv2.imshow("image", im)
    # cv2.waitKey(0)

    # # -1 is used to fill the object
    # cv2.drawContours(im, contours, -1, (0, 0, 255), -1)
    # cv2.imshow("image", im)
    # cv2.waitKey(0)


if __name__ == "__main__":
    main()
import cv2


def main(imagename):
    image = cv2.imread(imagename, 0)
    ret, thresh = cv2.threshold(image, 127, 255, 0)

    # calc those contours!
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # only compute for maximum contour
    maxarea, maxcnt = 0, 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > maxarea:
            maxarea = area
            maxcnt = contour

        outname = imagename.replace(".jpg", "_out.jpg")

        cv2.imwrite(outname, output)


if __name__ == "__main__":
    main()

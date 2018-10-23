# import the necessary packages
from PIL import Image
import pytesseract
import argparse
import cv2
import os
import numpy as np

def main():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="path to be OCR'd")

    args = vars(ap.parse_args())

    imgcnt = 0
    correct = 0

    extensions = [".tif",".jpg", ".png"]
    print("folder", args["folder"])
    for f in sorted(os.listdir(args["folder"])):
        ext = os.path.splitext(f)[1]
        if ext.lower() in extensions:
            imgcnt += 1
            res = ocr(args['folder'], f, args)
            if res:
                correct += 1
    print(correct, "correct out of", imgcnt)

def preprocess(gray):
    # check to see if we should apply thresholding to preprocess the
    # image
    # gray = cv2.threshold(gray, 0, 255,
    #     cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
     
    # # median blur to remove noise
    # # gray = cv2.medianBlur(gray, 3)

    # dilate to shrink the blobs and highlight the letters
    gray = keepCharsOnly(gray)
    gray = cv2.copyMakeBorder(gray,300,300,300,300,cv2.BORDER_CONSTANT,value=[255,255,255])

    # kernel = np.ones((3, 3), np.uint8)
    # gray = cv2.dilate(gray, kernel, iterations=3)
    #show(gray)
    return gray

def keepCharsOnly(inputImage):
    # Get the bounding boxes that we think are characters.
    # Get the extreme corners
    # Crop the image minX, maxY - maxX, minY

    minCharHeight = 80
    maxCharHeight = 150

    maxX = 0
    maxY = 0
    minX, minY = inputImage.shape

    #workingImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(inputImage, 127, 255, 0)
    thresh = cv2.bitwise_not(thresh)
    workingImage, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        currentContourBoundingRectExceeded = False
        currentContourMinRectExceeded = False
        currentContourRadiusExceeded = False

        #Characters tend to have:
        # w = 80
        # h = 140

        # get the bounding rect
        x, y, w, h = cv2.boundingRect(contour)  # tl
        # print("contour", cv2.boundingRect(contour))
        areaBoundingRect = w * h
        # logger.info("Bounding rectWidth %s, Height %s", w, h)
        if h > minCharHeight and h < maxCharHeight:
            maxX = x+w if x+w > maxX else maxX
            maxY = y+h if y+h > maxY else maxY
            minX = x if x < minX else minX
            minY = y if y < minY else minY
            # show(workingImage[minY:maxY,minX:maxX])

    #Crop image
    return inputImage[minY:maxY,minX:maxX]

def ocr(foldername, imgname, args):
    # load the example image and convert it to grayscale
    fullname = os.path.join(foldername, imgname)
    image = cv2.imread(fullname)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = preprocess(gray)

     
    # write the grayscale image to disk as a temporary file so we can
    # apply OCR to it
    filename = "{}.png".format(os.getpid())
    cv2.imwrite(filename, gray)
    # config = ('-l eng --oem --psm 3')
    config = ('-l eng --oem 1 -c tessedit_char_whitelist=jJ0123456789')
    # the temporary file
    text = pytesseract.image_to_string(Image.open(filename), config=config)
    os.remove(filename)
    text = text.replace(" ","")
    actual = imgname.split('_')[0]

    if actual == "blank":
        actual = ""

    if actual == text:
        print(actual,":",text, "match!")
        return True
    else:
        print(actual,":",text)
        show(gray)
        return False
     

def show(image, resize=800, windowname="image"):
    """
    Show the image in a window alsosave to file
    if s is pressed
    """

    global COUNTER
    if resize > 0:
        cv2.namedWindow(windowname, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(windowname, resize, resize)
    cv2.imshow(windowname, image)

    k = cv2.waitKey(0)
    if k == ord('s'):  # wait for 's' key to save and exit
        imagename = 'img{:04d}.png'.format(COUNTER)
        print("saving", imagename)
        cv2.imwrite(imagename, image)
        cv2.destroyAllWindows()
        COUNTER += 1
    elif k == ord('q'):
        exit()
    else:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

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
    gray = cv2.threshold(gray, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
     
    # median blur to remove noise
    # gray = cv2.medianBlur(gray, 3)

    # dilate to shrink the blobs and highlight the letters
    kernel = np.ones((2, 2), np.uint8)
    gray = cv2.dilate(gray, kernel, iterations=1)
    # show(gray)
    return gray

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
    config = ('-l eng --oem 0 -c tessedit_char_whitelist=jJ0123456789')
    # the temporary file
    text = pytesseract.image_to_string(Image.open(filename), config=config)
    os.remove(filename)
    actual = imgname.split('_')[0]

    if actual == "blank":
        actual = ""

    if actual == text:
        print(actual,":",text, "match!")
        return True
    else:
        print(actual,":",text)
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
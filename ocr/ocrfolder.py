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
    ap.add_argument("-f", "--folder", required=True,
        help="path to be OCR'd")
    ap.add_argument("-p", "--preprocess", type=str, default="thresh",
        help="type of preprocessing to be done")
    args = vars(ap.parse_args())

    imgcnt = 0
    correct = 0

    extensions = [".tif",".jpg", ".png"]
    print("folder", args["folder"])
    for f in os.listdir(args["folder"]):
        print("opening ", f)
        ext = os.path.splitext(f)[1]
        if ext.lower() in extensions:
            imgcnt += 1
            res = ocr(args['folder'], f, args)
            if res:
                correct += 1
    print(correct, "correct out of", imgcnt)

def ocr(foldername, imgname, args):
    # load the example image and convert it to grayscale
    fullname = os.path.join(foldername, imgname)
    image = cv2.imread(fullname)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
     
    # check to see if we should apply thresholding to preprocess the
    # image
    if args["preprocess"] == "thresh":
        gray = cv2.threshold(gray, 0, 255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
     
    # make a check to see if median blurring should be done to remove
    # noise
    elif args["preprocess"] == "blur":
        gray = cv2.medianBlur(gray, 3)

    #kernel = np.ones((3, 3), np.uint8)
    # gray = cv2.dilate(gray, kernel, iterations=2)
     
    # write the grayscale image to disk as a temporary file so we can
    # apply OCR to it
    filename = "{}.png".format(os.getpid())
    cv2.imwrite(filename, gray)
    # config = ('-l eng --oem --psm 3')
    config = ('-l eng --oem 1')
    # the temporary file
    text = pytesseract.image_to_string(Image.open(filename))
    os.remove(filename)
    actual = imgname.rstrip(".tif")
    actual = actual.rstrip('b')

    # show(gray)
    #show the output images
    # cv2.imshow("Image", image)
    # cv2.imshow("Output", gray)
    # cv2.waitKey(0)

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
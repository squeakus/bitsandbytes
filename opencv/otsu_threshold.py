import cv2
import sys
import glob
from tqdm import tqdm


def main(filename):

    if filename is None:
        jpgs = glob.glob("*.jpg")
        for jpg in tqdm(jpgs):
            otsu_thresh(jpg)
    else:
        otsu_thresh(filename)


def otsu_thresh(jpg):
    image = cv2.imread(jpg)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    outname = jpg.replace(".jpg", "_gray.jpg")
    cv2.imwrite(outname, gray)

    # thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # Otsu's thresholding
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    outname = jpg.replace(".jpg", "_thresh.jpg")
    cv2.imwrite(outname, thresh)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        print(f"running on file: {sys.argv[1]}")
        main(sys.argv[1])
    else:
        main(None)

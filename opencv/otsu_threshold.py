import cv2
import glob
from tqdm import tqdm


def main():

    jpgs = glob.glob("*.jpg")
    for jpg in tqdm(jpgs):
        image = cv2.imread(jpg, 0)
        # thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        # Otsu's thresholding
        ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        outname = jpg.replace(".jpg", "_thresh.jpg")
        cv2.imwrite(outname, thresh)


if __name__ == "__main__":
    main()
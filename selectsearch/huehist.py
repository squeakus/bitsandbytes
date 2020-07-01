import cv2
import argparse
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib import colors
from matplotlib.colors import hsv_to_rgb
from mpl_toolkits.mplot3d import Axes3D

def main():
    # Read args and load the input image
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
        help="path to the input image")
    args = vars(ap.parse_args())
    imagename = args["image"]
    outname = imagename.replace(".jpg", "_trimmed.jpg")
    image = cv2.imread(imagename)

    # lets plot the hsv colormap
    hsvcolors = []
    # iterate over saturation values
    for i in range(0,255):
        hsvrow = []
        #iterate over hue values
        for j in range(0,180):
            hsvrow.append([j,i,255])
        hsvcolors.append(hsvrow)
    hsvstrip = np.uint8(hsvcolors)
    hsvstrip = cv2.cvtColor(hsvstrip, cv2.COLOR_HSV2RGB)


    # hist, xbins, ybins = np.histogram2d(h.ravel(),s.ravel(),[180,256],[[0,180],[0,256]])
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist( [hsv], [0, 2], None, [180, 256], [0, 180, 0, 256] )
    plt.imshow(hist)
    plt.show()

    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    
    ax1 = plt.subplot(1, 2, 1)
    ax1.title.set_text('Hue Histogram')
    plt.hist(h.ravel(),18,[0,180]) 
    ax2 = plt.subplot(1, 2, 2)
    ax2.title.set_text('Hue / Saturation')
    plt.imshow(hsvstrip)
    plt.show()


if __name__=='__main__':
    main()
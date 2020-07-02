"""
Converts an RGB image and value to the HSV colorspace and then builds masks based on color ranges

If manually trying to find colors I found these the best:
https://alloyui.com/examples/color-picker/hsv.html
https://toolstud.io/color/rgb.php
Remember that most of these have a range 360,100,100 for HSV whereas opencv has 255 255 255.

Some sample ranges: 
light_orange = (1, 190, 200)
dark_orange = (18, 255, 255)

light_blue = (110, 50, 50)
dark_blue = (126,150,250)

light_white = (0, 0, 200)
dark_white = (145, 60, 255)

circuitboard green:[(57, 50, 0), (117, 255, 255)]
dark colors: [(0, 0, 0), (255, 255, 60)]
Based on the tutorial:
https://realpython.com/python-opencv-color-spaces/
"""

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
    outname = imagename.replace(".jpg", "_crop.jpg")
    image = cv2.imread(imagename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #hsv_range = compute_hsv_range([43,62,60], False)
    chip_range = [(0, 0, 0), (255, 255, 15)]
    green_range =  [(57, 50, 0), (117, 255, 255)]    
    chip_mask = compute_region(image, chip_range)
    green_mask = compute_region(image, green_range)
    #mask = chip_mask + green_mask
    # mask = 255 - mask
    result = extract_region(image, chip_mask)
    show_mask(result, chip_mask)

    contours = get_contours(chip_mask)
    contour_to_mask(image,contours[0])
    # #result = contour_mask(image, hsv_range)
    plt.imshow(result)
    plt.show()
#    cv2.imwrite(outname, cv2.cvtColor(result, cv2.COLOR_BGR2RGB))




def get_contours(image):
    # Use 5x5 kernel with erode and dialate to remove noise
    kernel = np.ones((5,5), np.uint8) 
    image = cv2.erode(image, kernel, iterations=2) 
    image = cv2.dilate(image, kernel, iterations=4) 
    image = cv2.erode(image, kernel, iterations=2) 
    result = extract_region(image, image)

    # Find contours:
    ret,thresh = cv2.threshold(image,127,255,0)
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return contours


def contour_to_mask(image, contour):
    height, width, depth = image.shape
    blank = np.zeros((height, width, depth), np.uint8)
    cv2.drawContours(blank, [contour],-1,(255,255,0), -1)
    hull = cv2.convexHull(contour)
    cv2.drawContours(blank, [hull], 0, (255,255,255), 3)
    print("convex hull has ",len(hull),"points")
    plt.imshow(blank)
    plt.show()




def chip_contour_mask(image, hsv_range):
    """
    Create a mask, smooth it, find the contour
    use the contour as a mask to extract everything in the mask.
    """
    #hsv_range = [(30, 0, 200), (145, 60, 255)]
    #check_color(hsv_range[0], hsv_range[1])
    mask = compute_region(image, hsv_range)



    # show_mask(result, mask)


    # M = cv2.moments(contours[0])
    # print(M)

    # use contours to make new mask


def show_color_plots(image):
    r, g, b = cv2.split(image)
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")

    pixel_colors = image.reshape((np.shape(image)[0]*np.shape(image)[1], 3))
    norm = colors.Normalize(vmin=-1.,vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()

    axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Red")
    axis.set_ylabel("Green")
    axis.set_zlabel("Blue")
    plt.show()

    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv_image)
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")

    axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Hue")
    axis.set_ylabel("Saturation")
    axis.set_zlabel("Value")
    plt.show()

def compute_hsv_range(rgbcolor, colorcheck=True):
    sensitivity = 30
    rgb_color = np.uint8([[rgbcolor]]) #here insert the bgr values which you want to convert to hsv
    hsv_color = cv2.cvtColor(rgb_color, cv2.COLOR_RGB2HSV)
    
    hsv_light = np.array([hsv_color[0][0][0] - sensitivity, 50, 50])
    if hsv_light[0] < 0:
        hsv_light[0] = 0
    hsv_dark = np.array([hsv_color[0][0][0] + sensitivity, 255, 255])

    if colorcheck:
        check_color(hsv_light, hsv_dark)
    print("HSV:", [hsv_light, hsv_dark])
    return [hsv_light, hsv_dark]

def check_color(hsv_range):
    hsv_light, hsv_dark = hsv_range[0], hsv_range[1]
    dark_square = np.full((10, 10, 3), hsv_light, dtype=np.uint8) / 255.0
    light_square = np.full((10, 10, 3), hsv_dark, dtype=np.uint8) / 255.0
    ax1 = plt.subplot(1, 2, 1)
    ax1.title.set_text('Light')
    plt.imshow(hsv_to_rgb(light_square))
    ax2 = plt.subplot(1, 2, 2)
    ax2.title.set_text('Dark')
    plt.imshow(hsv_to_rgb(dark_square))

    plt.show()

def compute_region(image, hsv_range):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv_image,hsv_range[0], hsv_range[1])
    return mask

def extract_region(image, mask):
    return cv2.bitwise_and(image, image, mask=mask)

def show_mask(result, mask):
    plt.subplot(1, 2, 1)
    plt.imshow(mask)
    plt.subplot(1, 2, 2)
    plt.imshow(result)
    plt.show()

if __name__=='__main__':
    main()
import cv2
import numpy as np
from matplotlib import pyplot as plt

image_name = "fruit_basket.jpg"
image = cv2.imread(image_name)
blue_channel = image_name.replace(".jpg", "_blue.jpg")
green_channel = image_name.replace(".jpg", "_green.jpg")
red_channel = image_name.replace(".jpg", "_red.jpg")

b, g, r = cv2.split(image)  # get b,g,r
x, y, z = np.shape(image)

red = np.zeros((x, y, z), dtype=int)
green = np.zeros((x, y, z), dtype=int)
blue = np.zeros((x, y, z), dtype=int)

for i in range(0, x):
    for j in range(0, y):
        blue[i][j][0] = image[i][j][0]
        green[i][j][1] = image[i][j][1]
        red[i][j][2] = image[i][j][2]


cv2.imwrite(blue_channel, blue)
cv2.imwrite(green_channel, green)
cv2.imwrite(red_channel, red)

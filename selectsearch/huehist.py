"""
Based on the tutorial:
https://realpython.com/python-opencv-color-spaces/
"""

import cv2
import argparse
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
args = vars(ap.parse_args())

# load the input image
image = cv2.imread(args["image"])

#a list of all the colorspaces 
# flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
# print(flags)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# plt.imshow(image)
# plt.show()

# r, g, b = cv2.split(image)
# fig = plt.figure()
# axis = fig.add_subplot(1, 1, 1, projection="3d")

# pixel_colors = image.reshape((np.shape(image)[0]*np.shape(image)[1], 3))
# norm = colors.Normalize(vmin=-1.,vmax=1.)
# norm.autoscale(pixel_colors)
# pixel_colors = norm(pixel_colors).tolist()

# axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
# axis.set_xlabel("Red")
# axis.set_ylabel("Green")
# axis.set_zlabel("Blue")
# plt.show()

hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
# h, s, v = cv2.split(hsv_image)
# fig = plt.figure()
# axis = fig.add_subplot(1, 1, 1, projection="3d")

# axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
# axis.set_xlabel("Hue")
# axis.set_ylabel("Saturation")
# axis.set_zlabel("Value")
# plt.show()


from matplotlib.colors import hsv_to_rgb
# color picket https://toolstud.io/color/rgb.php
# light_orange = (1, 190, 200)
# dark_orange = (18, 255, 255)
light_blue = (94,80,2)
dark_blue = (126,150,250)
light_blue = (110, 50, 50)
dark_blue = (130, 255, 255)

#blue
rgb_color = np.uint8([[[255, 0, 0]]]) #here insert the bgr values which you want to convert to hsv

#pcb_green
rgb_color = np.uint8([[[69, 71, 42]]]) #here insert the bgr values which you want to convert to hsv

hsv_color = cv2.cvtColor(rgb_color, cv2.COLOR_BGR2HSV)


hsv_lower = np.array([hsv_color[0][0][0] - 10, 50, 50])
hsv_upper = np.array([hsv_color[0][0][0] + 10, 255, 255])

print(hsv_color)
print(hsv_lower)
print(hsv_upper)

print("colorcheck")
lo_square = np.full((10, 10, 3), hsv_lower, dtype=np.uint8) / 255.0
do_square = np.full((10, 10, 3), hsv_upper, dtype=np.uint8) / 255.0
plt.subplot(1, 2, 1)
plt.imshow(hsv_to_rgb(do_square))
plt.subplot(1, 2, 2)
plt.imshow(hsv_to_rgb(lo_square))
plt.show()

print("showing masks")
mask = cv2.inRange(hsv_image,hsv_lower, hsv_upper)
result = cv2.bitwise_and(image, image, mask=mask)
plt.subplot(1, 2, 1)
plt.imshow(mask, cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(result)
plt.show()

# light_white = (0, 0, 200)
# dark_white = (145, 60, 255)

# mask_white = cv2.inRange(hsv_image, light_white, dark_white)
# result_white = cv2.bitwise_and(image, image, mask=mask_white)

# print("showing second masks")
# plt.subplot(1, 2, 1)
# plt.imshow(mask_white, cmap="gray")
# plt.subplot(1, 2, 2)
# plt.imshow(result_white)
# plt.show()

# final_mask = mask + mask_white
# print("showing final combo")

# final_result = cv2.bitwise_and(image, image, mask=final_mask)
# plt.subplot(1, 2, 1)
# plt.imshow(final_mask, cmap="gray")
# plt.subplot(1, 2, 2)
# plt.imshow(final_result)
# plt.show()

# blur = cv2.GaussianBlur(final_result, (7, 7), 0)
# plt.imshow(blur)
# plt.show()




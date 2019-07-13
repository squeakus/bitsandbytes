from scipy import misc

import matplotlib.pyplot as plt
import numpy as np
import math

import scipy.ndimage.filters as filters
import scipy.ndimage as ndimage

img = misc.imread('pentagon.png')

print('image shape: ', img.shape)

plt.imshow(img, )

plt.savefig("image.png",bbox_inches='tight')

plt.close()

img_shape = img.shape

x_max = img_shape[0]
y_max = img_shape[1]

theta_max = 1.0 * math.pi 
theta_min = 0.0

r_min = 0.0
r_max = math.hypot(x_max, y_max)

r_dim = 200 
theta_dim = 300

hough_space = np.zeros((r_dim,theta_dim))

for x in range(x_max):
    for y in range(y_max):
        if img[x,y,0] == 255: continue
        for itheta in range(theta_dim):
            theta = 1.0 * itheta * theta_max / theta_dim
            r = x * math.cos(theta) + y * math.sin(theta)
            ir = r_dim * ( 1.0 * r ) / r_max
            hough_space[int(ir),int(itheta)] += 1

plt.imshow(hough_space, origin='lower')
plt.xlim(0,theta_dim)
plt.ylim(0,r_dim)

tick_locs = [i for i in range(0,theta_dim,40)]
tick_lbls = [round( (1.0 * i * theta_max) / theta_dim,1) for i in range(0,theta_dim,40)]
plt.xticks(tick_locs, tick_lbls)

tick_locs = [i for i in range(0,r_dim,20)]
tick_lbls = [round( (1.0 * i * r_max ) / r_dim,1) for i in range(0,r_dim,20)]
plt.yticks(tick_locs, tick_lbls)

plt.xlabel(r'Theta')
plt.ylabel(r'r')
plt.title('Hough Space')

plt.savefig("hough_space_r_theta.png",bbox_inches='tight')

plt.close()

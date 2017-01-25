mport cv2
import numpy as np


img1 = cv2.imread('a.jpg', cv2.CV_LOAD_IMAGE_GRAYSCALE)
img2 = cv2.imread('b.jpg', cv2.CV_LOAD_IMAGE_GRAYSCALE)

diff = img1 - img2
cv2.imwrite('difference.jpg', diff)

#here is howto compute the match
difval = diff.sum()

import cv2
img1 = cv2.imread('test1.jpg', cv2.CV_LOAD_IMAGE_GRAYSCALE)
img2 = cv2.imread('test2.jpg', cv2.CV_LOAD_IMAGE_GRAYSCALE)
img3 = img1 -img2
cv2.imwrite('diff.jpg', img3)

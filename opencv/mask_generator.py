import cv2
import numpy as np

def main():
	im_gray = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)
	h,  w = im_gray.shape[:2]
	print('HW', h, w)
	thresh = 100
	im_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]

	kernel = np.ones((3,3),np.uint8)

	dilation = cv2.dilate(im_bw, kernel, iterations = 2)
	filled = dilation.copy()
	mask = np.zeros((h + 2, w + 2), np.uint8)
	cv2.floodFill(filled, mask, (h//2, w//2), 255)
	erosion = cv2.erode(filled, kernel, iterations = 3)
	erosion = cv2.bitwise_not(erosion)

	masked = cv2.bitwise_and(im_gray,im_gray, mask = erosion)

	cv2.imwrite('blackwhite.png', im_bw)
	cv2.imwrite('blackwhite1.png', dilation)
	cv2.imwrite('blackwhite2.png', filled)
	cv2.imwrite('blackwhite3.png', erosion)
	cv2.imwrite('blackwhite4.png', masked)
if __name__=='__main__':
	main()
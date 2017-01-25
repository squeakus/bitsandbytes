import cv2
import numpy as np
from matplotlib import pyplot as plt


def main():
    img1 = cv2.imread('half.jpg', cv2.CV_LOAD_IMAGE_GRAYSCALE)
    makegauss(img1, 0)
    #resized = cv2.resize(img1, (0,0), fx=pow(2,0), fy=pow(2,0))
    cv2.imwrite('bustimg1.jpg', img1)

    img2 = cv2.pyrDown(img1)
    makegauss(img2, 1)
    resized = cv2.resize(img2, (0,0), fx=pow(2,1), fy=pow(2,1))
    cv2.imwrite('bustimg2.jpg', img2)

    img3 = cv2.pyrDown(img2)
    makegauss(img3, 2)
    resized = cv2.resize(img3, (0,0), fx=pow(2,2), fy=pow(2,2))
    cv2.imwrite('bustimg3.jpg', img3)


    img4 = cv2.pyrDown(img3)
    makegauss(img4, 3)
    resized = cv2.resize(img4, (0,0), fx=pow(2,3), fy=pow(2,3))
    cv2.imwrite('bustimg4.jpg', img4)

def makegauss(img, scale):
    for i in range(10):
        name = "scale"+str(scale) + "diff"+str('%02d'%i)
        img = gaussdiff(img, name, scale)

def gaussdiff(img, name, scale):
    kernel = np.ones((5,5),np.float32)/25
    #blur = cv2.filter2D(img,-1,kernel)
    blur = cv2.GaussianBlur(img,(3,3),0)
    
    diff = img - blur
    big = cv2.resize(diff, (0,0), fx=pow(2,scale), fy=pow(2,scale))
    cv2.imwrite(name+'.jpg', big)

    return blur

if __name__=='__main__':
    main()
# plt.set_cmap('Greys')
# plt.subplot(131),plt.imshow(img1),plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(132),plt.imshow(blur),plt.title('Blurred')
# plt.xticks([]), plt.yticks([])
# plt.subplot(133),plt.imshow(diff),plt.title('difference')
# plt.xticks([]), plt.yticks([])
# plt.show()

# img2 = cv2.pyrDown(img1)
# cv2.imwrite('bust2.jpg',img2)

# img3 = cv2.pyrDown(img2)
# cv2.imwrite('bust3.jpg',img3)

# img4 = cv2.pyrDown(img3)
# cv2.imwrite('bust4.jpg',img4)

# img5 = cv2.pyrDown(img4)
# cv2.imwrite('bust5.jpg',img5)


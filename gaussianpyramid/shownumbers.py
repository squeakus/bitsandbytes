import cv2, sys
import numpy as np
from matplotlib import pyplot as plt

outfile = open('out.txt','w')
img1 = cv2.imread('bust3.jpg', cv2.CV_LOAD_IMAGE_GRAYSCALE)

for row in img1:
    for elem in row[:40]:
        outfile.write(str('%03d' % elem)+' ')
    outfile.write('\n')

outfile.close()


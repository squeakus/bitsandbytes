"""Read a numpy file and output an image."""
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def main(filename):
    depth_array = np.load(filename)
    print(depth_array.shape)
    for row in depth_array:
        print(np.max(row))


    img = Image.fromarray(depth_array, 'L')
    img.save('my.png')
    img.show()


if __name__=='__main__':
    if len(sys.argv) < 2:
        print("usage: python {} <filename>".format(sys.argv[0]))
        exit()
    main(sys.argv[1])

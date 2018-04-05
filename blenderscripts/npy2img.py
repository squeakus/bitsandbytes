"""Read a numpy file and output an image."""
import sys
import numpy as np
from PIL import Image


def main(filename):
    depth_array = np.load(filename)
    print(depth_array.shape)
    if np.max(depth_array) > 255:
        print("Values over 255! There is going to be truncations")
        depth_array = np.clip(depth_array, 0, 255)

    byte_array = depth_array.astype(np.uint8)
    img = Image.fromarray(byte_array, 'L')

    outfilename = filename.rstrip('npy')+'png'
    img.save(outfilename)
    # img.show()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("usage: python {} <filename>".format(sys.argv[0]))
        exit()
    main(sys.argv[1])

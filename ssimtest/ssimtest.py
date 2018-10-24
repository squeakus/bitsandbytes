import argparse
import numpy as np
import matplotlib.pyplot as plt

from skimage import data, img_as_float
from skimage.io import imread
from skimage.measure import compare_ssim as ssim

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--original", required=True,
        help="original image")
    ap.add_argument("-t", "--target", required=True,
        help="target image")

    args = vars(ap.parse_args())
    orig = imread(args['original'])
    orig = img_as_float(orig)
    targ = imread(args['target'])
    targ = img_as_float(targ)

    if not orig.shape == targ.shape:
        print("image sizes do not match!", orig.shape, targ.shape)
        exit()
        image_resized = resize(image, (image.shape[0] / 4, image.shape[1] / 4),
                       anti_aliasing=True)


    mse_error = mse(orig, targ)
    ssim_error = ssim(orig, targ, multichannel=True)
    print("mse:", mse_error, "ssim:", ssim_error)

def mse(x, y):
    return np.linalg.norm(x - y)

if __name__=='__main__':
    main()
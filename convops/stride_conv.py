import numpy as np
import cv2
arr = np.array([[1, 1, 1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2, 2, 2],
                [3, 3, 3, 3, 3, 3, 3],
                [4, 4, 4, 4, 4, 4, 4],
                [5, 5, 5, 5, 5, 5, 5],
                [6, 6, 6, 6, 6, 6, 6],
                [7, 7, 7, 7, 7, 7, 7]])

kernel = np.array([[1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1]])

kernel = np.ones((77,77))

def stride_conv(array, kernel, stride):
    width, height = array.shape
    kernw, kernh = kernel.shape
    beg = 0
    end = kernw
    final = []
    xlim = (width - (width % kernw)) - 1
    for i in range(0, xlim, stride):
        k = []

        ylim = (height - (height % kernh)) - 1
        for j in range(0, ylim, stride):
            k.append(np.sum(array[beg+i: end+i, beg+j:end+j] * (kernel)))
        final.append(k)

    return np.array(final)

image = cv2.imread('resized.png', 0)
normed = image / 255

result = stride_conv(normed, kernel, 77)
cv2.imwrite('out.png', result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

for row in result:
    print(row)

from scipy import signal
import numpy as np

arr = np.array([[2,3,7,4,6,2,9],
                [6,6,9,8,7,4,3],
                [3,4,8,3,8,9,7],
                [7,8,3,6,6,3,4],
                [4,2,1,8,3,4,6],
                [3,2,4,1,9,8,3],
                [0,1,3,9,2,1,4]])

arr2 = np.array([[1,1,1],
                 [1,1,1],
                 [1,1,1]])

result = signal.convolve2d(arr, arr2, mode='valid')
print(result)
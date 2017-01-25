import numpy as np

f = open("IMG_1416.mat", "r")
a = np.fromfile(f, dtype=np.uint32)
for elem in a:
    print elem

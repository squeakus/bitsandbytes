import ctypes as ct
import numpy as np

ct.cdll.LoadLibrary('libgrn.so')
grnlib = ct.CDLL('libgrn.so')

test_array = np.array([[2,3,4,5,6],[1,2,3,4,5]])
test_array = test_array.astype(np.float32)

c_array = test_array.ctypes.data_as(ct.POINTER(ct.c_float))

#py_arr = [4,5,6,7,8,9]
#test_array = (ct.c_float * len(py_arr))(*py_arr)

grnlib.passarray2(2,5, c_array)

for item in test_array:
    print item

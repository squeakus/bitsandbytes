import ctypes
ctypes.cdll.LoadLibrary('libgrn.so')
grnlib = ctypes.CDLL('libgrn.so')

grnlib.test()
moo = [0,1,2,3,4,2]
result =  (ctypes.c_int * len(moo))()
arr = (ctypes.c_int * len(moo))()
arr[:] = moo
print "result", list(result)
grnlib.print_array(arr,result, len(moo))

print "result", list(result)

import ctypes
ctypes.cdll.LoadLibrary('libdist.so')
distlib = ctypes.CDLL('libdist.so')
print str(distlib.distance(1000,1000,1000,0,0,0))

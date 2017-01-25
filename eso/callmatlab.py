from mlabwrap import mlab
import numpy
from numpy import pi, subtract, sin, cos

# mlab.plot([1,2,3],'-o')
# xx = numpy.arange(-2*pi, 2*pi, 0.2)
# mlab.surf(subtract.outer(sin(xx),cos(xx)))
# from numpy import *
# mlab.surf(subtract.outer(sin(xx),cos(xx)))
# mlab.lookfor('singular value')
# #have to specify if there are multiple args
# U, S, V = mlab.svd([[1,2],[1,3]], nout=3)
# mlab.top(60,20,0.5,3.0,1.5)

moo = numpy.array([1,2,3,4,5])
print "numpyarray", moo
mlab.testfunc(moo)

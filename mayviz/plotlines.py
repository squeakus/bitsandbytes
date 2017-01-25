import numpy
from mayavi.mlab import *

def test_plot3d():
    """Generates a pretty set of lines."""
    n_mer, n_long = 6, 11
    pi = numpy.pi
    dphi = pi / 1000.0
    phi = numpy.arange(0.0, 2 * pi + 0.5 * dphi, dphi)
    mu = phi * n_mer
    #x = numpy.cos(mu) * (1 + numpy.cos(n_long * mu / n_mer) * 0.5)
    #y = numpy.sin(mu) * (1 + numpy.cos(n_long * mu / n_mer) * 0.5)
    #z = numpy.sin(n_long * mu / n_mer) * 0.5
    x = numpy.array([0,1,2,3,4,5,6,7,8,9])
    y = numpy.array([0,1,2,3,4,5,2,7,3,6])
    z = numpy.array([0,1,2,3,4,4,3,2,1,0])


    l = plot3d(x, y, z, tube_radius=0.025, colormap='Spectral')
    return l

test_plot3d()
show()

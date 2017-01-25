import pylab
import numpy as np

concentrations = np.loadtxt('MackeyGlass_t17.txt')
print concentrations
print "max", max(concentrations), "min", min(concentrations)
pylab.plot(concentrations[0:200])
pylab.show()

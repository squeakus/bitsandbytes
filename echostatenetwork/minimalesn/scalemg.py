import numpy as np
import pylab

unscaled = np.loadtxt('MackeyGlass_t17.txt')
maxval = max(unscaled)
minval = min(unscaled)

#normalise between zero and 1
scaled = []
for point in unscaled:
    newpoint = (point - minval) / (maxval - minval)
    scaled.append(newpoint)

outfile = open('mgscaled.txt', 'w')
for point in scaled[:1000]:
    outfile.write(str(point)+"\n")
pylab.plot(scaled[:1000])
pylab.show()

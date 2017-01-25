#!/usr/bin/env python
# a bar plot with errorbars

#!/usr/bin/env python
import numpy.numarray as na

from pylab import *

labels = ["Low MOEA Fitness", "High MOEA Fitness", "No Preference"]
data =   [55.9, 36.84 , 7.26]

xlocations = na.array(range(len(data)))+0.5
width = 0.5
bar(xlocations, data, color=('r','g','b'), width=width)
#yticks(range(0, 100))
xticks(xlocations+ width/2, labels)
xlim(0, xlocations[-1]+width*2)
ylim(0,100)
ylabel("Percentage of preference")
title("Preference Survey Results")
gca().get_xaxis().tick_bottom()
gca().get_yaxis().tick_left()

show()







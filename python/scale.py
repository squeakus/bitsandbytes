import random
def scale(x, minrange, maxrange):
    #scale from 0,1 to whatever range
    newval = ((maxrange - minrange) * x) + minrange
    print "original", x, "new", newval

def normalize(value, old_range):
    #reduce from range to 0,1
    if value < old_range[0]: value = old_range[0]
    if value > old_range[1]: value = old_range[1]
    
    normalized = (float(value) -old_range[0]) / (old_range[1] - old_range[0])
    return normalized

    
scale(0,-100,100)
scale(1,-100,100)
scale(0.5,-100,100)

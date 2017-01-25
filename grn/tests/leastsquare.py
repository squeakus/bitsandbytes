import pylab
import numpy as np


def least_square(x,y):
    """linear regression based on the least squares formula"""
    x_squared = np.square(x)
    meanx = np.mean(x)
    meany = np.mean(y)
    xy = x * y

    #slope formula
    numerator = sum(xy) - (sum(x)*sum(y))/len(x) 
    denominator = sum(x_squared) - (sum(x)*sum(x))/len(x)
    slope = numerator / denominator 

    #Y intercept (b = y - mx)
    yintercept = meany-slope*meanx

    bestfit = []
    for i in range(len(x)):
        bestfit.append(slope*float(i)+yintercept)

    return bestfit

def normalize(value, maxval, minval):
    #scale between zero and one
    norm = (value - minval) / (maxval - minval)
    return norm
    
y = np.loadtxt('randomwalk.dat')
#scale the input
y = y * 0.4
x = range(0,len(y))
maxfit = len(y)

constsig = [np.mean(y)] * len(y)
nosig = np.array([0]*len(y))
best = least_square(x,y)

linearfit = maxfit - sum(abs(y-best))
minfit = maxfit - sum(abs(y-nosig))
constfit = maxfit - sum(abs(y-constsig))

print "maximumfit", maxfit
print "minimumfit", minfit
print "best constant fitness", constfit, "norm", normalize(constfit, maxfit, minfit)
print "best linear fitness", linearfit, "norm", normalize(linearfit, maxfit, minfit)

normed = {}
for i in np.arange(0,61,10):
    offsetsig = np.roll(y,i)
    offsetfit = maxfit - sum(abs(y-offsetsig))
    normed[str(i)] = normalize(offsetfit,maxfit,minfit)

for key in normed:
    if normed[key] < 0: normed[key] = 0

print normed
pylab.plot(y)
pylab.plot(constsig)
pylab.plot(best)
pylab.show()

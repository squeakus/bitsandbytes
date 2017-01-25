from numpy import *
from pylab import *

x = loadtxt('randomwalk.dat')
# for i in arange(0,61,10):
#     clf()
#     y = roll(x, i)
plot(x)
#     plot(y)
xlabel("timesteps")
title('Random Walk Data ')
#legend(["input","target"])
savefig('randomwalk.png')


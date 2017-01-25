import numpy as np
import pylab as P

mu, sigma = 200, 25
x = mu + sigma*P.randn(10000)

x = mu + sigma*P.randn(1000,3)
n, bins, patches = P.hist(x, 10, normed=1, histtype='barstacked')

#n, bins, patches = P.hist(x, 10, normed=1, histtype='bar',
#                          color=['r', 'g','b'],
#                          label=['Crimson','Burlywood', 'Chartreuse'])

P.legend()
print x
P.show()

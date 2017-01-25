import numpy as np
import timeit

#x = range(1000)
# or 
x = np.random.standard_normal(1000)

def pure_sum():
        return sum(x)

def numpy_sum():
	return np.sum(x)

n = 10000

t1 = timeit.timeit(pure_sum, number = n)
print 'Pure Python Sum:', t1
t2 = timeit.timeit(numpy_sum, number = n)
print 'Numpy Sum:', t2

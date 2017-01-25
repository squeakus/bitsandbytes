import pylab as P

P.plot([1,2,3], [1,2,3], 'go-', label='line 1', linewidth=2)
P.plot([1,2,3], [1,4,9], 'go--',  label='line 2', linewidth=2)
P.axis([0, 4, 0, 10])
P.legend()
P.figure()

P.plot([1,2,3], [9,2,3], label='line 1', linewidth=2)
P.plot([1,2,3,4,5,6], [1,4,9,4,5,6],  label='line 2', linewidth=2)
P.axis([0, 10, 0, 10])
P.legend()
P.figure()

P.show()

import random, pylab

x = [0.5]
biggest = 0
for i in range(1,1000):
    # if random.random() > 0.5:
    #     x.append(x[i-1]+0.01)
    # else:
    #     x.append(x[i-1]-0.01)

    increment = ((2*random.random())-1) / 50
    if abs(increment) > biggest: biggest = increment
    newval = x[i] + increment
    x.append(newval)
    print x
    if x[i] > 1: x[i] = 1
    if x[i] < 0: x[i] = 0
print x
print "biggest", biggest
pylab.ylim(0,1)
pylab.plot(x, linewidth=0.2)

pylab.savefig('out.pdf')
pylab.show()

outfile = open("randomwalk.txt",'w')
for point in x:
    outfile.write(str(point)+'\n')
outfile.close()

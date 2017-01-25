import numpy, math, pylab
sine = [(math.sin(i) + 1)/2
        for i in numpy.arange(0,4*math.pi, 0.1)]

offsetfitnesses = []
lowest = 126
for i in range(len(sine)+1):
    offset = i
    fitness = 0

    #100.8
    for index in range(len(sine)):
        offidx = (index - offset) % len(sine)
        x = 1 - abs(sine[offidx]-sine[index])
        fitness += 1 - abs(sine[offidx]-sine[index])
    if fitness < lowest: lowest = fitness
    offsetfitnesses.append(fitness) 
print "L",lowest
pylab.xlabel("offset")
pylab.ylabel("fitness")
pylab.plot(offsetfitnesses)
pylab.savefig("sineoffset.pdf")

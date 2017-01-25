#!/usr/bin/python                                                              

from matplotlib import pyplot
import time

initPop = 0.5
growthRate = 0.5
runs = 40
generations = 100

def nextPop(pop,rate):
    nextPop = pop*rate*(1-pop)
    return nextPop



for run in range(runs):
    result = []
    print "run", run, "rate", growthRate
    pop = nextPop(initPop,growthRate)
    for i in range(generations):
        pop = nextPop(pop,growthRate)
        result.append(pop)
    pyplot.figure(run)
    pyplot.plot(range(0,generations), result)
    pyplot.ylim([-1,1])
    pyplot.xlabel('generation')
    pyplot.ylabel('population size')
    pyplot.title("Rate "+str(growthRate))
    pyplot.grid(True)
    #pyplot.savefig('simple_plot')
    #pyplot.close('all')
    growthRate += .1
pyplot.show()


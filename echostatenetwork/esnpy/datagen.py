import numpy as np
import pylab, math

def mackeyglass(timesteps, alpha=0.2, beta=10, 
                gamma=0.1, tau=17, trunc = 300):
    """Returns a time series based on the concentration 
    of white blood cells. It is a tunably chaotic time series.
    Tau of 7 is mild, 17 reasonable, and 30 is wild.
    The first 300 timesteps are truncated as it does not initially
    oscillate"""
    series = np.zeros(timesteps+trunc)
    series[0] = 1.2

    for i in range(1, timesteps+trunc-1):
        # only calculate t when there is enough data points
        if (i - tau) < 0:
            prev_conc = 0
        else:
            prev_conc = series[i - tau]

        nom = alpha * prev_conc
        denom = 1 + (np.power(prev_conc, beta))
        decay = (gamma*series[i])
        series[i+1] = series[i] + ((nom / denom) - decay)
    print "a", len(series)
    print "b", len(series[trunc:])
    return series[trunc:]

def sine(timesteps, freq=10):
    """Boring old sine wave"""
    series = np.zeros(timesteps)
    for i in range(timesteps):
        series[i] = math.sin(i/freq)
    return series


def varsine(timesteps, freq=5.0):
    """Boring old sine wave"""
    series = np.zeros(timesteps)
    offset = 0
    for i in range(timesteps):
        if i > 0:
            if i % 300 == 0:
                freq = 10.0
                offset = 30

            if i % 600 == 0:
                freq = 20.0
                offset = 60
                
        series[i] = math.sin(offset + (i/freq))
        print series[i] * 0.5
    return series

    
def feedback(timesteps, init_conc, prod_rate, decay):
    """Playing with decays, not quite sure
    what I was trying to do here"""
    series = np.zeros(timesteps)
    series[0] = init_conc

    for i in range(1, timesteps):
        series[i] = prod_rate - (decay * series[i-1])

    return series

def logistic(timesteps, init_pop=0.001, growth_rate=3.9):
    """Based on the logistic differential equation, a simulation
    of population based on feedback from the previous pop,
    unlike the malthusian approach which just increases exponentially.
    It exhibits non linear behavior for high pop rates"""
    series = np.zeros(timesteps)
    series[0] = init_pop

    for i in range(1,timesteps):
        newpop = series[i-1] * growth_rate*(1-series[i-1])
        series[i] = newpop
    return series

def main():

    # result = mackeyglass(500)
    # print len(result)
    # pylab.figure(1).clear()
    # pylab.plot(result)

    # result = sine(900, 5.0) * 0.5
    # pylab.figure(2).clear()
    # pylab.plot(result)

    # result = logistic(100, 0.001, 3.5)
    # pylab.figure(3).clear()
    # pylab.plot(result)
    result = varsine(900)
    pylab.figure(4).clear()
    pylab.plot(result)
    pylab.show()

if __name__ == '__main__':
    main()

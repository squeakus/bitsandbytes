import random, datagen
import esn as ESN
from esn import mse
from numpy import hstack, empty
def main():
    """Read data, train on data, test on data, BAM""" 
    data = datagen.sine(300, 5.0) * 0.5
    data = hstack((data, datagen.sine(300, 10.0) * 0.5))
    data = hstack((data, datagen.sine(300, 20.0) * 0.5))
    inputs = empty(300)
    inputs.fill(0.0)
    tmp = empty(300)
    tmp.fill(0.5)
    inputs = hstack((inputs, tmp))
    tmp.fill(1.0)
    inputs = hstack((inputs, tmp))

    besttrain = {'val':1000,'idx':0}
    besttest = {'val':1000,'idx':0}
    
    for i in range(1000):
        random.seed(i)
        esn = ESN.ESN(input_size=1, hidden_size=100, output_size=1)
        train_out = esn.train(data, inputs)
        test_out = esn.test(data, inputs)
        trainmse = mse(train_out, data[100:])
        testmse = mse(test_out, data)

        if trainmse < besttrain['val']:
            besttrain['val'] = trainmse
            besttrain['idx'] = i
            print "newbesttrain", trainmse

        if testmse < besttest['val']:
            besttest['val'] = testmse
            besttest['idx'] = i
            print "newbesttest", testmse

    #calculate mean square error
    print "Train MSE", besttrain 
    print "Test MSE", besttest

if __name__ == '__main__':
    main()

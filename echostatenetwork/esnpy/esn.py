from random import random, seed
from numpy import zeros, ones, empty, linalg, nditer, tanh
from numpy import dot, array, loadtxt, arctanh, vstack, hstack
from sklearn import linear_model
import graph, datagen

# offset the weights with input bias?
# do the weights look right?
# spectral radius sometimes increases the values
# different initial weights for hidden and back layers
# why do we need the arctan of the target?
seed(244)
        
def main():
    """Read data, train on data, test on data, BAM""" 
    esn = ESN(input_size=1, hidden_size=100, output_size=1)
    test_damping(esn, 200)

    #create inputs and training data
    #data = datagen.mackeyglass(900, trunc=300) * 0.5
    data = loadtxt('varsine.dat')
    inputs = empty(300)
    inputs.fill(0.0)
    tmp = empty(300)
    tmp.fill(0.5)
    inputs = hstack((inputs, tmp))
    tmp.fill(1.0)
    inputs = hstack((inputs, tmp))
    graph.plot_2d([inputs,data],"data")
    #train and test the esn
    train_out = esn.train(data, inputs)
    test_out = esn.test(data, inputs)
    graph.plot_2d([train_out, data[100:900]], "train_out")
    graph.plot_2d([test_out, data], "test_out")
    #plot residuals
    train_res = residual(train_out, data[100:])
    test_res = residual(test_out, data)
    graph.plot_2d([train_res],"train_residual")
    graph.plot_2d([test_res],"test_residual")

    #calculate mean square error
    print "Train MSE", mse(train_out, data[100:])
    print "Test MSE", mse(test_out, data)

class ESN:
    """An echo state network implementation based on code from 
    Fred Cummins and Mantas Lukoevius"""
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        self.acts = None # stores the activations during the run
        self.weights = dict()
        self.alpha = 1.0
        self.spec_scale = 0.8 #why this additional scaling

        # input connected to every node in hidden layer 
        self.weights['input'] = zeros((hidden_size, input_size))
        init_weights(self.weights['input'], 1.0)

        # initialise the recursive layer sparsely (< 10% connected)
        self.weights['hidden'] = zeros((hidden_size, hidden_size))
        init_weights(self.weights['hidden'], 0.05, 0.4)
            
        # every backward connection gets a weight
        self.weights['back'] = zeros((hidden_size, output_size))
        init_weights(self.weights['back'], 1.0)

        # compute spectral radius of recursive layer (max eigenvalue) 
        spec_rad  = max(abs(linalg.eig(self.weights['hidden'])[0]))
        if spec_rad == 0:
            print "reservoir too small"
            exit()

        #divide the weights by spectral radius and then scale 
        print 'Spectral radius:', spec_rad
        self.weights['hidden'] =  self.weights['hidden'] / spec_rad
        self.weights['hidden'] *= self.spec_scale
        
    def train(self, target, inputs, trim=100):
        """Calculate weights between hidden layer and output layer for
        a given time series, uses pseudo-inverse training step"""
        acts = zeros((len(target), self.hidden_size))
        summed_acts = []
        
        #create initial state for the hidden nodes
        for i in range(len(acts[0])):
            acts[0][i] = (random()*2)-1

        # create the activations
        for i in range(1, len(target)):
            # turn target into array
            targ, inp = array([target[i-1]]), array([inputs[i-1]])
            # dotting target with back weights as the teacher signal
            activation = tanh(dot(acts[i-1],self.weights['hidden'])+
                              dot(self.weights['back'], targ)+
                              dot(self.weights['input'], inp))
            # leaky integrator: prev state effects current state
            acts[i] = ((1-self.alpha) * acts[i-1])
            acts[i] += self.alpha * activation

        #trim out the initial 100 activations as they are unstable
        target = target[trim:]
        inputs = inputs[trim:]
        acts = acts[trim:, :]

        #store activations and plot
        self.acts = acts
        graph.plot_2d(acts.T, "training_activations")

        #add the inputs to the activations
        acts = vstack((acts.T,inputs)).T
        
        # Pseudo-inverse to train the output and setting weights
        tinv = arctanh(target)
        clf = linear_model.RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
        #clf = linear_model.Ridge(alpha=0.5)
        #clf = linear_model.LassoCV()
        clf.fit(acts, tinv)
        self.weights['out'] = linear_model.ridge_regression(acts, tinv,alpha=.15)
        self.weights['out'] = clf.coef_
        #self.weights['out'] = linalg.lstsq(acts, tinv)[0]
        #residual = dot(acts, self.weights['out']) - tinv

        graph.bar_plot(self.weights['out'], "weights")

        # checking the output against previous activations
        train_out = []
        for act in acts:
            output = tanh(dot(act, self.weights['out']))
            train_out.append(output)
        return train_out

    def test(self, testdata, inputs):
        """run the esn, see what happens"""
        acts = zeros((len(testdata)+1, self.hidden_size))
        acts[0] = self.acts[-1] # setting initial activation
        test_out = []

        for i in range(1, len(acts)):
            output = tanh(dot(hstack((acts[i-1], inputs[i-1])), 
                              self.weights['out']))
            test_out.append(output)
            # compute next set of activations
            back, inp = array([output]), array([inputs[i-1]])
            activation = tanh(dot(acts[i-1], self.weights['hidden'])+
                              dot(self.weights['back'], back)+
                              dot(self.weights['input'], inp))
            acts[i] = (1-self.alpha) * acts[i-1] # decay
            acts[i] += self.alpha * activation 
        return test_out

#Utility methods
def mse(out, target):
    """returns mean square error"""
    return sum([pow((target[i]-out[i]), 2) 
                for i in range(len(target))])/len(target)

def residual(out, target):
    """returns timeseries of the error""" 
    res = []
    for idx, targ in enumerate(target):
        res.append(targ - out[idx])
    return res

def init_weights(weights, probability, scaling=1.0):
    """ Sparsely populate network and scale the weights"""
    for node in nditer(weights, op_flags=['readwrite']):
        if random() < probability:
            if scaling > 0:
                node[...] = ((random() * 2) - 1) * scaling
            else:
                if random > 0.5:
                    #I really hate this bit and don't know why it is here
                    node[...] = scaling
                else:
                    node[...] = -scaling


def test_damping(esn, iterations):
    """Checking if the network stabilises"""
    acts = zeros((iterations, esn.hidden_size))
    #set them to random initial activations
    for i in range(len(acts[0])):
        acts[0][i] = (random()*2)-1

    # lets see if it dampens out
    for i in range(1, iterations):
        acts[i] = ((1-esn.alpha) * acts[i-1])
        acts[i] += esn.alpha * tanh(dot(acts[i-1], 
                                         esn.weights['hidden']))
    graph.plot_2d(acts.T, "damped_activations")
            
if __name__ == '__main__':
    main()

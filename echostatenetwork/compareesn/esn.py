from random import random, seed
from numpy import zeros, linalg, divide, nditer, tanh
from numpy import dot, array, loadtxt, arctanh
import scipy.sparse.linalg as scialg
import graph, datagen
import numpy
# offset the weights with input bias?
# do the weights look right?
# spectral radius sometimes increases the values
# different initial weights for hidden and back layers
# why do we need the arctan of the target?
#seed(0)

def mse(out, target):
    """returns mean square error"""
    return sum([pow((target[i]-out[i]),2) 
                for i in range(len(target))])/len(target)

def residual(out, target):
    """returns timeseries of the error""" 
    res = []
    for idx, targ in enumerate(target):
        res.append(targ - out[idx])
    return res

def init_weights(weights, probability, scaling=1.0):
    """ Assign a weight between -1,1 probabalistically"""
    for node in nditer(weights, op_flags=['readwrite']):
        if random() < probability:
            node[...] = ((random() * 2) - 1) * scaling

class ESN:
    """An echo state network implementation based on code from 
    Fred Cummins and Mantas Lukoevius"""
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        self.acts = None # stores the activations during the run
        self.weights = dict()
        self.alpha = 1
        self.spec_scale = 0.8 #why this additional scaling
        
        #inputs not considered yet
        #self.weights['input'] = zeros((hidden_size,input_size+1))

        # initialise the recursive layer sparsely(10%)
        #self.weights['hidden'] = zeros((hidden_size, hidden_size))
        #init_weights(self.weights['hidden'], 0.05, 0.4)

        self.weights['hidden'] = array([[ 0. ,  0. ,  0. , -0. , -0. , -0. ,  0. , -0. ,  0. ,  0. , -0.4, -0. , -0. ,  0. , -0. , -0. ,  0. ,  0. ,  0. , -0. ],
       [ 0. ,  0. ,  0. , -0. ,  0. ,  0. ,  0. ,  0. , -0. ,  0. ,  0. ,
         0. , -0. , -0. , -0. , -0. ,  0.4, -0. ,  0. ,  0. ],
       [ 0. ,  0. ,  0. , -0. , -0. ,  0. , -0. , -0. ,  0. , -0. ,  0. ,
         0. ,  0. ,  0. , -0. , -0. , -0. ,  0. , -0. , -0. ],
       [ 0. ,  0. , -0. , -0. , -0. , -0. , -0. , -0. ,  0. ,  0. , -0. ,
         0. , -0. ,  0. , -0. , -0. , -0. , -0. , -0. , -0. ],
       [ 0.4, -0. ,  0. , -0. , -0. ,  0. , -0. ,  0. ,  0. , -0. ,  0.4,
        -0. ,  0. , -0. , -0. ,  0. , -0. , -0. , -0. , -0. ],
       [ 0. , -0. , -0. , -0. ,  0. ,  0. ,  0. , -0. ,  0. , -0. , -0. ,
        -0.4,  0. ,  0. ,  0. , -0. , -0. , -0. ,  0. , -0. ],
       [-0. ,  0. , -0. ,  0. , -0. , -0. ,  0. , -0. ,  0. , -0. , -0. ,
        -0. ,  0. , -0.4,  0. , -0. , -0. , -0. , -0. ,  0. ],
       [ 0. , -0. ,  0. , -0. ,  0. ,  0. ,  0. ,  0. , -0. , -0. , -0. ,
        -0. ,  0. ,  0. , -0. , -0. , -0.4, -0. ,  0. ,  0. ],
       [-0. , -0. ,  0. , -0. , -0. ,  0.4,  0. ,  0. ,  0. , -0. , -0. ,
        -0. , -0. , -0. ,  0. ,  0. ,  0. ,  0. , -0. ,  0.4],
       [-0. , -0. ,  0. ,  0. , -0. ,  0. , -0. , -0. , -0. ,  0. , -0. ,
        -0. , -0. , -0. ,  0. ,  0. , -0. , -0. , -0. ,  0. ],
       [ 0. ,  0. ,  0. , -0. ,  0. , -0. , -0. , -0.4,  0. , -0. , -0. ,
         0. , -0. ,  0. ,  0.4,  0. , -0. , -0. ,  0. ,  0. ],
       [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. , -0. , -0.4,  0. , -0. ,
        -0. ,  0. ,  0. ,  0. ,  0. , -0. ,  0. ,  0. , -0. ],
       [-0. , -0. ,  0. ,  0. , -0. ,  0. ,  0. , -0. ,  0. , -0. ,  0. ,
        -0. ,  0. , -0. ,  0.4, -0. ,  0.4,  0. , -0. ,  0. ],
       [ 0. ,  0. ,  0. ,  0. ,  0. , -0. ,  0. ,  0. ,  0. ,  0. , -0. ,
        -0.4,  0. ,  0. ,  0. ,  0. , -0. , -0. , -0. , -0. ],
       [-0. ,  0. ,  0. , -0. ,  0. ,  0. ,  0. , -0. , -0. ,  0. ,  0. ,
         0. ,  0. , -0. , -0. , -0. ,  0. , -0. ,  0. ,  0. ],
       [ 0. ,  0. ,  0. , -0. , -0. , -0. , -0. ,  0. , -0. , -0. , -0. ,
         0. ,  0. ,  0. ,  0. ,  0. , -0. , -0. , -0. ,  0. ],
       [-0. ,  0. ,  0. , -0. ,  0. , -0. , -0.4, -0. , -0. , -0. , -0. ,
         0. ,  0. , -0. , -0. , -0. , -0. ,  0.4, -0. , -0. ],
       [-0. , -0. ,  0. , -0. ,  0. , -0. ,  0. , -0. ,  0. , -0. , -0. ,
        -0. ,  0. , -0. , -0. ,  0. , -0. ,  0. , -0. , -0. ],
       [-0. , -0. , -0. ,  0. ,  0. , -0. , -0. ,  0. ,  0. , -0. ,  0. ,
        -0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. , -0. ],
       [ 0. ,  0. ,  0.4,  0. ,  0. , -0. , -0. ,  0. , -0.4, -0. ,  0. ,
        -0. ,  0. , -0. ,  0. , -0. ,  0. ,  0. ,  0.4,  0. ]])

        # every backward connection gets a weight
        # self.weights['back'] = zeros((hidden_size, output_size))
        # init_weights(self.weights['back'], 1.0)

        self.weights['back'] = array([[-1,-1,1,1,-1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,1,1,1]])
        
        # compute spectral radius of recursive layer (max eigenvalue) 
        spec_rad  = max(abs(linalg.eig(self.weights['hidden'])[0]))
        if spec_rad == 0:
            print "reservoir too small"
            exit()
            
        # scale the radius and then scale the weights
        print 'Spectral radius:', spec_rad
        #spec_rad *= self.spec_scale
        #self.weights['hidden'] = divide(self.weights['hidden'], spec_rad)
        
    def test_damping(self, iterations):
        """Checking if the network stabilises"""
        numpy.set_printoptions(precision=4,linewidth=2000)
        acts = zeros((iterations, self.hidden_size))

        acts[0] = [0.949029602769840,-0.195694633071382,-0.737442723592159,0.449467774003339,0.799036414715420,-0.658597370832685,-0.913942617588595,-0.0416815501470254,-0.812127430489660,0.300099663301922,0.904555077842031,-0.0845746012151277,0.0737612940120569,-0.867025708396808,-0.0122598328349043,-0.164918055240550,-0.415485899463317,-0.420672124930247,0.507691993375378,-0.806408576103972]

        # lets see if it dampens out
        for i in range(1, iterations):
            #acts[i] = ((1-self.alpha) * acts[i-1])
            #acts[i] += self.alpha * tanh(dot(acts[i-1], 
            #                                 self.weights['hidden']))
            product = dot(acts[i-1], self.weights['hidden'].T)
            #            print "product",product
            #            print "tanh", tanh(product)
        graph.plot_2d(acts.T, "damped_activations")

        
    def train(self, target, trim=100):
        """Calculate weights between hidden layer and output layer for
        a given time series, uses pseudo-inverse training step"""
        acts = zeros((len(target), self.hidden_size))
        summed_acts = []

        acts[0] = [0.949029602769840,-0.195694633071382,-0.737442723592159,0.449467774003339,0.799036414715420,-0.658597370832685,-0.913942617588595,-0.0416815501470254,-0.812127430489660,0.300099663301922,0.904555077842031,-0.0845746012151277,0.0737612940120569,-0.867025708396808,-0.0122598328349043,-0.164918055240550,-0.415485899463317,-0.420672124930247,0.507691993375378,-0.806408576103972]
        
        # create the activations
        self.weights['back'] = self.weights['back'].T
        for i in range(1,len(target)):
            # turn target into array, should it be -1?
            t = array([target[i-1]])

            # dotting target with back weights as the teacher signal
            activation = tanh(dot(acts[i-1], self.weights['hidden'].T)+
                              dot(self.weights['back'],t))
            #decay activation by alpha
            acts[i] = ((1-self.alpha) * acts[i-1])
            acts[i] += self.alpha * activation
            #print i-1, acts[i-1]

        #trim out the initial activations as they are unstable
        target = target[trim:]
        acts = acts[trim:,:]

        #store activations and plot
        self.acts = acts

        graph.plot_2d(acts.T, "sample_activations")
        #print "last",acts[-1]
        # Pseudo-inverse to train the output and setting weights
        tinv = arctanh(target)
        pinv = linalg.pinv(acts)
        aconj = acts.conjugate()
        tconj = tinv.conjugate()
        pinvaconj = linalg.pinv(aconj)
        
        numpy.set_printoptions(precision=4,linewidth=2000)

        # print "Pseudo inverse"
        #for row in aconj:
        #     print row
        
        print "acts",aconj.shape,"targ",tconj.shape
        #self.weights['out'] = dot(pinv, tinv)
        #self.weights['out'] = dot(pinvaconj,tconj)
        self.weights['out'] = linalg.lstsq(aconj,tconj)[0]
        print "\nnumpy lstsq", self.weights['out']

        #self.weights['out'] = linalg.lstsq(aconj,tconj)[0]
        #self.weights['out'] = scialg.lsqr(aconj,tconj)[0]

        #self.weights['out'] = array([-2.62751866452261,0.911724638334178,0,0,0,-9.47053869171588,0,0,-3.90801067449960,0,7.05096786026327,11.5512880240272,7.35498461772589,-0.221961288369604,0,0,0.681830006046591,0,0,-0.458162995895149])

        print "\nnew weights", self.weights['out']
        graph.bar_plot(self.weights['out'], "weights")

        # checking the output against previous activations
        train_out = []
        for act in acts:
            output = tanh(dot(act, self.weights['out']))
            train_out.append(output)
        return train_out

    def test(self, testdata):
        """run the esn, see what happens"""
        acts = zeros((len(testdata)+1, self.hidden_size))
        acts[0] = self.acts[-200] # setting initial activation
        test_out = []

        for i in range(1,len(testdata)+1):
            output = tanh(dot(acts[i-1], self.weights['out']))
            test_out.append(output)
            # compute next set of activations
            back = array([output])
            activation = tanh(dot(acts[i-1], self.weights['hidden'].T)+
                                    dot(self.weights['back'],back))
            acts[i] = (1-self.alpha) * acts[i-1] # decay
            acts[i] += self.alpha * activation
        return test_out

def main():
    # input not used yet
    esn = ESN(input_size=1, hidden_size=20, output_size=1)
    esn.test_damping(200)
    #gendata = datagen.sine(600) * 0.5
    data = loadtxt('sine2.dat')
    train_data = data[:300]
    test_data = data[:50]
    
    train_out = esn.train(train_data)
    test_out = esn.test(test_data)
    graph.plot_2d([train_out, data[100:300]], "train_out")
    graph.plot_2d([test_out, data[:50]], "test_out")

    #plot residuals
    train_res = residual(train_out, train_data[100:])
    test_res = residual(test_out, test_data)
    graph.plot_2d([train_res],"train_residual")
    graph.plot_2d([test_res],"test_residual")

    #calculate mse
    train_error = mse(train_out, train_data[100:])
    test_error = mse(test_out, test_data)

    print "Train MSE", train_error
    print "Test MSE", test_error

if __name__ == '__main__':
    main()

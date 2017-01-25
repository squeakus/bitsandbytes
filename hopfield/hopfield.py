import numpy as np
import random

def train(node_a, node_b):
    """nodes define the connection strength"""
    weight = (node_a) * (node_b)
    return weight

def solvefornode(testinput, weightarray):
    """node value is function of the input vector and the weights"""
    return np.dot(testinput,weightarray)

def main():
    #training = np.array([[-1,-1,-1,-1,-1]])
    training = np.array([[-1,1,1,-1,1],[1,-1,1,-1,1],[1,-1,1,1,-1]])
    testinput = np.array([-1,-1,-1,1,1])
    solved = False
    
    # train on examples, sum the weight arrays
    weightarray = np.zeros((5,5))
    
    for idx, row in enumerate(training):
        for i in range(len(row)):
            for j in range(len(row)):
                if i != j:
                    weightarray[i][j] += train(row[i],row[j])

    #calculate average
    weightarray = np.divide(weightarray, training.shape[0])
    print "Weights after training:\n", weightarray

    """There are three possible approaches to updating the nodes
    synchronously:
    update all nodes and then look for changes.
    e.g. [1,2,3,4,5],[1,2,3,4,5]
    Randomly:
    hit nodes until they all have stopped changing.
    [0,2,2,1,3,2,2,1,1,1,3,2,2,2,3,4]
    shuffle:
    better than random, but still stochastic
    [2,3,1,5,4],[3,1,2,4,5]
    """
    #random.seed(0)
    rowlist = [0,1,2,3,4]
    
    while not solved:
        random.shuffle(rowlist)
        print "reshuffling", rowlist
        #Only finish when all have stabilised
        stabilised = np.array([False]*5)

        #solve in a single timestep
        #print "total solve",np.dot(testinput,weightarray)
    
        for idx in rowlist:
            signal = solvefornode(testinput, weightarray[idx])
            msg = ''
            # depending on signal, change the node value
            if signal >= 0:
                if testinput[idx] != 1: 
                    msg = "change 1"
                    testinput[idx] = 1
                else:
                    stabilised[idx] = True
            else:
                if testinput[idx] != -1: 
                    msg =  "change -1"
                    testinput[idx] = -1
                else:
                    stabilised[idx] = True

            print "idx", idx, testinput, stabilised, msg
            if np.all([stabilised]):
                print "finished", idx, testinput, stabilised
                solved = True

if __name__ == '__main__':
    main()
    # print train(0,0)
    # print train(1,0)
    # print train(1,1)

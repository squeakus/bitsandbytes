import numpy as np
import random, graph, pylab

#random.seed(0)
def simple_grn(graphname):
    size = 5
    iterations = 10
    edges = np.zeros([size,size])
    nodes = np.empty([size])
    result_array = [[] for i in range(size)]

    """
    Initialise the nodes to have equal concentrations and
    initialise the weight array with vals between -1,1
    with stepsize 1
    """
    nodes.fill(1.0/size)
    weightvals  = np.arange(-1, 1.1, 0.1)
    for x in np.nditer(edges, op_flags=['readwrite']):
        #x[...] = float(random.randint(-1,1)) / (size)
        x[...] = round(random.choice(weightvals),2)

    print "nodes\n",nodes, "\nedges\n", edges

    for itr in range(iterations):
        #nudge a node halfway through
        #if itr == (iterations/2):
        #    nodes[0] = nodes[0] * 0.2
           
        #calculate concentration changes for the nodes
        change = np.empty(size)
        for i in range(len(nodes)):
            change[i] = np.dot(nodes,edges[i])

        # alter concs and prevent it from going below zero
        print "iter:",itr, "conc:", nodes, "change:", change 
        nodes = np.add(nodes,change)
        nodes[nodes<0] = 0.000001

        #normalize
#        total = sum(nodes)
#        nodes = np.divide(nodes,total)

        #record concentrations
        for i in range(size):
            result_array[i].append(nodes[i])

    graph.plot_2d(result_array,"graph"+str(graphname))

if __name__ == '__main__':
    for i in range(100):
        simple_grn(i)

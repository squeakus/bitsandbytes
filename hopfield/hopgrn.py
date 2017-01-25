import numpy as np
import random, graph, pylab, dotgraph

def normalize(value, new_range, old_range):
    this function is broken
    if value > old_range[1]:
        value = old_range[1]
    normalized = ((new_range[0] + (value - old_range[0])
                   * (new_range[1] - new_range[0]))
                  / old_range[1] - old_range[0])
    return normalized

def hop_grn(async, size, iterations):
    total_change = 0
    edges = np.zeros([size,size])
    nodes = np.empty([size])
    result_array = [[] for i in range(size)]

    """node can be either [-1,1]"""
    for node in np.nditer(nodes, op_flags=['readwrite']):
        node[...] = int(random.choice([-1,1]))
    
    """Two methods for initializing weights, binary or range"""
    weightvals  = np.arange(-1, 1.1, 0.1)
    for edge in np.nditer(edges, op_flags=['readwrite']):
        
        #binary weights
        #edge[...] = float(random.randint(-1,1))

        #range weighting
        
        edge[...] = round(random.choice(weightvals),2)
        
    print "node:\n",nodes, "\nedges:\n",edges
    nodelist = range(len(nodes))
    for itr in range(iterations):
        change = 0
        signals = np.empty(size)
        random.shuffle(nodelist)
        
        for i in nodelist:
            signals[i] = np.dot(nodes,edges[i])

            #sync/async update techniques
            if async:
                if signals[i] >= 0:
                    if nodes[i] == -1: change += 1
                    nodes[i] = 1
                else:
                    if nodes[i] == 1: change += 1
                    nodes[i] = -1

        if not async:
            for i in range(len(signals)):
                if signals[i] >= 0:
                    if nodes[i] == -1: change += 1
                    nodes[i] = 1
                else:
                    if nodes[i] == 1: change += 1
                    nodes[i] = -1

        #collect results
        for i in range(size):
            result_array[i].append(nodes[i])
        total_change += change
        #print "iter:",itr, "vals:", nodes, "signals", signals,"change",change
        #dotgraph.draw_graph("iter"+str(itr), nodes, edges)
    if change == 0:
        return True, total_change
    else:
        return False, total_change


if __name__ == '__main__':
    labels, data = [], []
    stabilised, total_change = hop_grn(False, 3, 100)
    #stuff for testing different sizes
    # for size in range(2,11):
    #     stab_count = 0
    #     for i in range(10000):
    #         stabilised, total_change = hop_grn(False, size, 100)

    #         if stabilised:
    #             stab_count += 1
    #     print "size:",size,"stabilised", stab_count
    #     labels.append(str(size))
    #     data.append(stab_count)
        
    # xlocations = np.array(range(len(data)))+0.5
    # width = 0.5
    # pylab.bar(xlocations, data, width=width)
    # pylab.xticks(xlocations+ width/2, labels)
    # pylab.xlim(0, xlocations[-1]+width*2)
    # pylab.ylim(0,10000)
    # pylab.xlabel("size of network")
    # pylab.ylabel("number that stabilised")
    
    # Pylab.title("Synchronous update")
    # pylab.gca().get_xaxis().tick_bottom()
    # pylab.gca().get_yaxis().tick_left()

    # pylab.show()

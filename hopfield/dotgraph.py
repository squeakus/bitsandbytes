import pydot, random
import numpy as np

def draw_graph(name, nodes, edges, weights=True):
    size = nodes.shape[0]
    graph = pydot.Dot('grn', graph_type='digraph') 
    nodelist = []

    for i in range(size):
        if nodes[i] > 0:
            nodelist.append(pydot.Node(i,label='+1', 
                                       style="filled",
                                       fillcolor='red'))
        else:
            nodelist.append(pydot.Node(i,label='-1',
                                       style="filled",
                                       fillcolor='blue'))
    for node in nodelist:
        graph.add_node(node)

    for row in range(size):
        for col in range(size):
            weight = str(round(edges[row][col],2))
            if not weights:
                edge = pydot.Edge(nodelist[row],nodelist[col])
            else:
                edge = pydot.Edge(nodelist[row],
                                  nodelist[col],
                                  label=weight)
            graph.add_edge(edge)
    graph.write_jpg(name + ".jpg")

def main():
    for size in range(2,3):
        nodes = np.empty(size)
        edges = np.zeros([size,size])
        
        """node can be either [-1,1]"""
        for node in np.nditer(nodes, op_flags=['readwrite']):
            node[...] = int(random.choice([-1,1]))
        
        for edge in np.nditer(edges, op_flags=['readwrite']):
            edge[...] = float(random.randint(-1,1))
            
        draw_graph("weight"+str(size), nodes, edges)


if __name__ == '__main__':
    main()
    

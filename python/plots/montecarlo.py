import random as R
import ctypes
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.axes
from math import sqrt

# Using C library to compute distance
ctypes.cdll.LoadLibrary('libdist.so')
DISTLIB = ctypes.CDLL('libdist.so')

#returns euclidean distance between two points
def distance(pt_1, pt_2):
    """Uses C library to calculate euclidean distance"""
    dist = DISTLIB.distance(int(pt_1[0]), int(pt_1[1]), int(pt_1[2]),
                            int(pt_2[0]), int(pt_2[1]), int(pt_2[2]))
    return dist

def euclidean_distance(p, q):
    return sqrt(sum([(p[i] - q[i]) **2 for i in range(len(p))]))

def create_graph(node_count):
    tmp_graph = []
    for i in range(node_count):
        point = [R.randint(0,10), R.randint(0,10),R.randint(0,10)]
        tmp_graph.append(point)
    return tmp_graph

def total_distance(graph_1, graph_2):
    total_dist = 0
    for node_1 in graph_1:
        min_dist = 1000 
        for node_2 in graph_2:
            dist = euclidean_distance(node_1, node_2)
            if dist < min_dist:
                min_dist = dist
        total_dist += min_dist
    return total_dist


def check_triangle(graph_a, graph_b, graph_c):
    a_b = total_distance(graph_a, graph_b)
    a_c = total_distance(graph_a, graph_c)
    c_b = total_distance(graph_c, graph_b)
    a_c_b = a_c + c_b
    difference = a_b - a_c_b

    if difference > 0.0005:
        fig = plt.figure()
        #ax = Axes3D(fig)

        print difference
        print "problem! ab", a_b,"ac",a_c,"cb",c_b,"acb", a_c_b
        print "A:", graph_a
        print "B:", graph_b
        print "C:", graph_c

        for i, symb,col in [(graph_a, 'o','b'),
                            (graph_b, '^','r'),
                            (graph_c, 'o','g')]:
           
            xs, ys, zs = [], [], []
            for point in i:
                xs.append(point[0])
                ys.append(point[1])
                zs.append(point[2])
            #ax.scatter(xs, ys, zs,s=100, c=col, marker=symb)
            plt.scatter(xs, ys,s=100, c=col, marker=symb)

        plt.show()


for i in range(1,5):
    print "checking", i
    for j in range(1000):
        graph_a = create_graph(i)
        graph_b = create_graph(i)
        graph_c = create_graph(i)
        check_triangle(graph_a, graph_b, graph_c)

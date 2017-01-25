import networkx as nx
import matplotlib.pyplot as plt
from geometry import *
import graph

g = graph.graph()
g.add_unique_node([0,0,0], "test")
g.add_unique_node([5,0,0], "test")
g.add_unique_node([0,5,0], "test")
g.add_unique_node([5,5,0], "test")

g.nearest_node(2,0,0)
g.nearest_node(0,3,0)
g.nearest_node(3,3,0)

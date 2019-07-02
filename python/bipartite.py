import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from networkx.drawing.nx_agraph import graphviz_layout
from networkx.algorithms import bipartite

adj = np.array([[0,0,0,0,0,1,1,1,1,1],
				[0,0,0,0,0,1,1,1,1,1],
				[0,0,0,0,0,1,1,1,1,1],
				[0,0,0,0,0,1,1,1,1,1],
				[0,0,0,0,0,1,1,1,1,1],
				[0,0,0,0,0,0,0,0,0,0],
				[0,0,0,0,0,0,0,0,0,0],
				[0,0,0,0,0,0,0,0,0,0],
				[0,0,0,0,0,0,0,0,0,0],
				[0,0,0,0,0,0,0,0,0,0]])

G = nx.bipartite.gnmk_random_graph(5, 5, 25, seed=123)
top = nx.bipartite.sets(G)[0]
pos = nx.bipartite_layout(G, top)

#G = nx.from_numpy_matrix(adj, nx.DiGraph())
# nx.write_gexf(G, "test.gexf")
nx.draw_networkx(G, pos, with_labels=True)
plt.show()
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from networkx.drawing.nx_agraph import graphviz_layout


adj = np.array([[0,1,0,0,1,0,0,0,0,0],
				[0,0,1,0,0,1,0,0,0,0],
				[0,0,0,1,0,0,1,0,0,0],
				[0,0,0,0,0,0,0,1,0,0],
				[0,0,0,0,0,0,0,0,1,1],
				[0,0,0,0,0,0,0,0,0,0],
				[0,0,0,0,0,0,0,0,0,0],
				[0,0,0,0,0,0,0,0,0,0],
				[0,0,0,0,0,0,0,0,0,0],
				[0,0,0,0,0,0,0,0,0,0]])
G = nx.from_numpy_matrix(adj, nx.DiGraph())
pos=graphviz_layout(G, prog='dot')
nx.draw_networkx(G,pos, with_labels=True)
plt.show()
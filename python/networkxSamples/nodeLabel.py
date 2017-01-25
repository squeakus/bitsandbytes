import networkx as nx
import pylab

G=nx.dodecahedral_graph()
labels=nx.draw_networkx_labels(G,pos=nx.spring_layout(G))
pylab.show()

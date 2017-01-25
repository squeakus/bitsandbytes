import networkx as nx 
import pylab 
G = nx.Graph() 
G.add_edge(1, 2, weight=2) 
G.add_edge(2, 3, weight=10) 
pos=nx.graphviz_layout(G,prog="twopi")
#pos=nx.spring_layout(G) 
# version 1 
pylab.figure(1) 
nx.draw(G,pos) 
# use default edge labels 
nx.draw_networkx_edge_labels(G,pos) 
# version 2 
pylab.figure(2) 
nx.draw(G,pos) 
# specifiy edge labels explicitly 
edge_labels=dict([((u,v,),d['weight']) 
                 for u,v,d in G.edges(data=True)]) 
nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels) 
# show graphs 
pylab.show() 

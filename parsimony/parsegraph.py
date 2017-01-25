import networkx as nx
import matplotlib.pyplot as plt

class Parsegraph(nx.Graph):
    def __init__(self, *args, **kwargs):
        super(Parsegraph, self).__init__(*args, **kwargs)
        self.time_slices = [] # array for holding graphs


    def word_query(self, noun):
        for node_idx in self.nodes():
            if self.node[node_idx]["word"] == noun:
                return node_idx

    def get_edges(self, node):
        edge_list = []
        for edge in self.edges():
            if node in edge:
                edge_list.append(self[edge[0]][edge[1]])
        return edge_list
                
    def add_figure(self,figure):
        for key in figure.keys():
            self.add_node(figure["idx"])
            #add all the other info
            if not key == "idx":
                self.node[figure["idx"]][key] = figure[key]

    def update(self, figures, edge_list):
        for figure in figures:
            if not figure["idx"] in self.nodes():
                #print "adding new node"
                self.add_figure(figure)
        self.remove_edges_from(self.edges())
            
        for edge in edge_list:
            self.add_edge(edge["fig_a"], edge["fig_b"], label=edge["label"])

        #add a copy of yourself to the time slices
        new_graph = self.copy()
        self.time_slices.append(new_graph)        


    # output all time slices in sequence
    def generate_graphs(self, figures):
        for idx, time_graph in enumerate(self.time_slices):
            print "generating time frame", idx
            self.generate_graph(time_graph, figures, idx)

    def generate_graph(self, time_graph, figures, name):
            plt.clf()
            #pos = nx.spring_layout(time_graph)
            pos = nx.draw_graphviz(time_graph, prog="twopi")
            # pos = nx.circular_layout(time_graph)
            # #pos = nx.spectral_layout(time_graph)
            # nx.draw_networkx_nodes(time_graph, pos, node_color='b',
            #                        node_size=1000, alpha=0.8)

            # edge_labels=dict([((u,v,),d['label'])
            #                   for u,v,d in time_graph.edges(data=True)])
            
            # node_labels=dict([(fig["idx"],fig["label"])
            #                   for fig in figures])

            # nx.draw_networkx_edges(time_graph, pos, edge_labels=edge_labels)
            # nx.draw_networkx_edge_labels(time_graph, pos,
            #                              edge_labels=edge_labels)

            # nx.draw_networkx_labels(time_graph, pos,node_labels)
            plt.axis('off')
            name = str(name)+".png"
            
            plt.savefig(name)             

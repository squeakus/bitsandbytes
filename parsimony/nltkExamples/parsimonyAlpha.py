# Regular Expression parser that tags the generated tokens
# borrowed from: 
#http://stackoverflow.com/questions/526469/practical-examples-of-nltk-use

import nltk
import networkx as nx
import matplotlib.pyplot as plt

def noun_query(parse_graph, noun):
    for node_idx in parse_graph.nodes():
        if parse_graph.node[node_idx]["noun"] == noun:
            print "found ", noun, "node index:", node_idx
    return node_idx

def label_query(parse_graph, label):
    for node_idx in parse_graph.nodes():
        if parse_graph.node[node_idx]["noun"] == label:
            print "found ", label, "node index:", node_idx
    return node_idx

def get_edges(parse_graph, node):
    for edge in parse_graph.edges():
        if node in edge:
            print "edge:",parse_graph[edge[0]][edge[1]]

def get_figure_name(figures, name):
    for figure in figures:
        if figure["name"] == name:
            return figure


if __name__ == '__main__':
    source_identifiers = ["from"]
    target_identifiers = {"into":"in", "to":"at"}
    verbs = ["went"]
    sentence = "the ball went from the tree into the house" 
    count = {"verbs":0} # a dictionary of counters
    figures = []
    verb = None
    source_identifier = None
    target_identifier = None
    actor = None
    source = None
    target = None
    time_slices = [] #list for holding the timeslice graphs

    #master ground figure
    master_ground = {"idx":len(figures),  #id on the graph
                     "name":"MG",         # Fig designation
                     "word":"MASTER",     # "the ball"
                     "label":r'$MG$',     # label for graph
                     "role": "MG"}        # Actor, source, etc
    
    parse_graph = nx.Graph()
    parse_graph.add_node(len(figures),label="MG", noun=None)
    figures.append(master_ground)

    #FIND THE FIGURES
    tokens = sentence.split(" ")
    for i in range(len(tokens)):
        if tokens[i] == "the":
            fig_word = tokens[i]+" "+tokens[i+1]
            fig_name =  "F"+ str(len(figures))
            fig_label = r'$'+fig_name+fig_word+'$'

            #figure dictionary
            figure = {"idx":len(figures),
                      "name":fig_name,
                      "word":fig_word,
                      "label":fig_label,
                      "role":None} 

            # rewrite figures into sentence
            sentence = sentence.replace(fig_word, fig_name)
            
            print "adding node:", len(figures) 
            parse_graph.add_node(len(figures),label="none",noun=tokens[i+1]);
            #attaching to master ground
            parse_graph.add_edge(0, len(figures), label="in")
            #add figure to fig dict
            figures.append(figure)

    # query graph methods
    noun_query(parse_graph, "ball")
    get_edges(parse_graph, 1)

    # FIND THE PLAYERS
    tokens = sentence.split(" ")

    for i in range(len(tokens)):        
        if tokens[i] in verbs:            
            actor_fig = get_figure_name(figures, tokens[i-1])
            actor_fig["role"] = "actor"
            print "actorFig:", actor_fig
            verb = tokens[i]

        if tokens[i] in source_identifiers:
            source_fig = get_figure_name(figures, tokens[i+1])
            source_fig["role"] = "source"
            source_identifier = tokens[i]

        if tokens[i] in target_identifiers.keys():
            target_identifier = tokens[i]
            target_fig = get_figure_name(figures, tokens[i+1])
            target_fig["role"] = "target"
                                         

    act_idx = actor_fig["idx"]
    src_idx = source_fig["idx"]
    trg_idx = target_fig["idx"]

    #find them on the graph
    new_graph = parse_graph.copy()
    time_slices.append(new_graph)
    
    print "act",act_idx,"src",src_idx,"trg",trg_idx

    parse_graph.add_edge(act_idx, src_idx, label="at")
    parse_graph.remove_edge(0,act_idx)

    new_graph = parse_graph.copy()
    time_slices.append(new_graph)

    parse_graph.add_edge(act_idx, src_idx, label="near")

    new_graph = parse_graph.copy()
    time_slices.append(new_graph)
    
    parse_graph.remove_edge(act_idx, src_idx)
    parse_graph.add_edge(act_idx, trg_idx, label="near")

    new_graph = parse_graph.copy()
    time_slices.append(new_graph)

    parse_graph.add_edge(act_idx, trg_idx,
                         label=target_identifiers[target_identifier])

    new_graph = parse_graph.copy()
    time_slices.append(new_graph)
    

    for idx, time_graph in enumerate(time_slices):
#        for check_edge in time_graph.edges():
#            if act_idx in check_edge:
#                print check_edge.get_edge_data()
        
#    parse_graph.add_edge(act_idx, trg_idx, label="near")
#    parse_graph.node[act_idx]["label"] = "ACTOR" 
#    parse_graph.node[src_idx]["label"] = "SOURCE"
#    for node in parse_graph.nodes_iter():
#    print parse_graph.node[node]


        plt.clf()
        pos=nx.spring_layout(time_graph)
        #pos=nx.graphviz_layout(time_graph)
        nx.draw_networkx_nodes(time_graph, pos, node_color='b',
                           node_size=1000, alpha=0.8)

        edge_labels=dict([((u,v,),d['label'])
                      for u,v,d in time_graph.edges(data=True)])

        nx.draw_networkx_edges(time_graph,pos,edge_labels=edge_labels)
        nx.draw_networkx_edge_labels(time_graph,pos,edge_labels=edge_labels)  
        
        node_labels = {}
        for figure in figures:
            node_labels[figure["idx"]] = figure["label"]
        
        print node_labels

        nx.draw_networkx_labels(time_graph, pos,node_labels)
        plt.axis('off')
        img_name = str(idx)+".png"
        plt.savefig(img_name)             

# Regular Expression parser that tags the generated tokens
# borrowed from: 
#http://stackoverflow.com/questions/526469/practical-examples-of-nltk-use

import nltk
import parsegraph
import networkx as nx
import matplotlib.pyplot as plt

def get_figure_name(figures, name):
    for figure in figures:
        if figure["name"] == name:
            return figure

def change_edge_label(edge_list, fig_a, fig_b, label):
    found = False
    for edge in edge_list:
        pts = [edge["fig_a"], edge["fig_b"]]
        if fig_a in pts and fig_b in pts:
            found = True
            edge["label"] = label
    if not found: print "Could not find edge to change label"
    return edge_list

def switch_edges(edge_list, fig_a, fig_b, fig_c, label):
    for edge in edge_list:
        pts = [edge["fig_a"], edge["fig_b"]]
        if fig_a in pts and fig_b in pts:
            edge_list.remove(edge)

        if fig_b in pts and fig_c in pts:
            print "edge already exists"

    edge = {"fig_a":fig_b,
            "fig_b":fig_c,
            "label":label}
    edges.append(edge)
    
    return edge_list

def print_edges(edge_list):
    for edge in edge_list:
        print "A:",edge["fig_a"],"B:",edge["fig_b"],"=",edge["label"]



if __name__ == '__main__':
    source_identifiers = ["from"]
    target_identifiers = {"into":"in", "to":"at"}
    verbs = ["went"]
    sentence = "the ball went from the tree into the house" 

    parse_graph = parsegraph.Parsegraph() # graph object for storing changes
    count = {"verbs":0} # a dictionary of counters
    figures = []
    edges = []
    verb = None
    source_id = None
    target_id = None

    #master ground figure
    master_ground = {"idx":len(figures),  #id on the graph
                     "name":"MG",         # Fig designation
                     "word":"MASTER",     # "the ball"
                     "label":r'$MG$',     # label for graph
                     "role": "MG"}        # Actor, source, etc
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
            figures.append(figure)

            # connect to master ground
            edge = {"fig_a":figures[0]["idx"],
                    "fig_b":figure["idx"],
                    "label":"in"}
            edges.append(edge)

            #reformat sentence
            sentence = sentence.replace(fig_word, fig_name)

    parse_graph.update(figures, edges)

    # testing query graph methods
    #print "node index for ball:", parse_graph.word_query("the ball")
    #print "number of edges connected to MG", parse_graph.get_edges(0)

    # FIND THE PLAYERS
    tokens = sentence.split(" ")
    for i in range(len(tokens)):        
        if tokens[i] in verbs:            
            actor_fig = get_figure_name(figures, tokens[i-1])
            actor_fig["role"] = "actor"
            verb = tokens[i]

        if tokens[i] in source_identifiers:
            source_fig = get_figure_name(figures, tokens[i+1])
            source_fig["role"] = "source"
            source_id = tokens[i]

        if tokens[i] in target_identifiers.keys():
            target_fig = get_figure_name(figures, tokens[i+1])
            target_fig["role"] = "target"
            target_id = tokens[i]
                                         
    act_idx = actor_fig["idx"]
    src_idx = source_fig["idx"]
    trg_idx = target_fig["idx"]

    #time slice 1
    edges = switch_edges(edges, 0, act_idx, src_idx, "at")
    parse_graph.update(figures, edges)

    # time slice 2
    edges = change_edge_label(edges, act_idx, src_idx, "near")
    parse_graph.update(figures, edges)

    # time slice 3
    edges = switch_edges(edges, src_idx, act_idx, trg_idx, "near")
    parse_graph.update(figures, edges)

    # time slice 4
    #edges = change_edge_label(edges, act_idx, trg_idx,target_identifiers[target_id])
    edges = change_edge_label(edges, act_idx, trg_idx, "at")
    parse_graph.update(figures, edges)

    # spit out time slices
    parse_graph.generate_graphs(figures)     

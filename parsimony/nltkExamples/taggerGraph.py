# Regular Expression parser that tags the generated tokens
# borrowed from: 
#http://stackoverflow.com/questions/526469/practical-examples-of-nltk-use

import nltk
import networkx as nx
import matplotlib.pyplot as plt




def tag(text, tokenizer, tagger):
    tokenized = tokenizer.tokenize(text)    
    tagged = tagger.tag(tokenized)
    return tagged

def create_graph(tagged, parse_graph):
    current_frame = 0
    labels = {}
    for idx, word in enumerate(tagged):
        if word[1] == None: labels[idx] = r'$MOO$'
        else: labels[idx] = r'$'+word[1]+'$'
        current_frame += 1
        parse_graph.add_node(idx,label=word[1])
        parse_graph.add_edge(idx,idx-1)
        frame_name = "output/%03d.png" % current_frame
        print "frame name", frame_name

       #generate a lovely graph
        plt.clf()
        pos=nx.spring_layout(parse_graph)
        nx.draw_networkx_nodes(parse_graph, pos, node_color='b',
                           node_size=1000, alpha=0.8)
        nx.draw_networkx_edges(parse_graph,pos,width=2.0,alpha=0.5)
        nx.draw_networkx_labels(parse_graph, pos,labels)
        plt.axis('off')
        plt.savefig(frame_name)    
        #plt.show() # display


if __name__ == '__main__':
    sample_file = open('sample_text_2.txt', 'r')
    TEXT = sample_file.read()
    TOKENIZER = nltk.tokenize.RegexpTokenizer(r'\w+|[^\w\s]+')
    TAGGER = nltk.UnigramTagger(nltk.corpus.brown.tagged_sents())
    TAGGED = tag(TEXT, TOKENIZER, TAGGER)
    print TAGGED    

    PARSE_GRAPH = nx.Graph()
    create_graph(TAGGED, PARSE_GRAPH)

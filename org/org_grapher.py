import networkx as nx
import matplotlib.pyplot as plt
import csv
import pygraphviz
from networkx.drawing.nx_agraph import write_dot
try:
    import pygraphviz
    from networkx.drawing.nx_agraph import graphviz_layout
except ImportError:
    try:
        import pydot
        from networkx.drawing.nx_pydot import graphviz_layout
    except ImportError:
        raise ImportError("This example needs Graphviz and either "
                          "PyGraphviz or pydot")


def main():
    G = nx.Graph()
    org = read_org('Bob.csv')
    # ann = read_org("AnnKelleher.csv")
    #org = read_org("PeterCharvat.csv")

    labels = {}
    
    for uid in org.keys():
        G.add_node(uid)
        labels[uid] = org[uid]['name']
        if org[uid]['parent'] is not '':
            G.add_edge(org[uid]['parent'], uid)

    # #nx.draw_circular(G, node_size=10)
    # plt.figure(figsize=(30, 30))
    # pos = graphviz_layout(G, prog='twopi', args='')
    # nx.draw(G, pos, node_size=5, alpha=0.5, node_color="blue", with_labels=False)
    # plt.axis('equal')
    # plt.show()


    #for idx, emp in enumerate(org):
    #
    #     G.add_node(emp['uid'])
    #     labels['uid'] = emp['name']
    #     if emp['parent'] is not '':
    #         G.add_edge(emp['parent'], emp['uid'])            

    # print("Nodes:", G.number_of_nodes())
    # print("Edges:", G.number_of_edges())
    # nodes = list(G.nodes)
    
    # use digraph for this stuff
    # reports = []
    # singles = 0
    # for node in nodes:

    #     staff = len(list(G.successors(node)))
    #     if staff > 0:
    #         reports.append(staff)
    #     else:
    #         singles += 1

    #     if staff > 30:
    #         print(node, " has ", staff)
    # print("singles: ", singles)
    #plt.hist(reports, bins=35)
    #plt.show()

    # depths_graph = nx.shortest_path_length(G, nodes[0])
    # depth = min(depths_graph.values())
    # # print("depth", depth)
    # # diameter = nx.algorithms.distance_measures.diameter(G)
    # # print
    # # max_clique_size = nx.algorithms.clique.graph_clique_number(G)
    # # print(depth, diameter, max_clique_size)
    
    nx.write_gexf(G, "test2.gexf")
    print("written gxf")
    # #pos=nx.spring_layout(G)
    # print("spring")
    # nx.draw(G)
    # print("draw")
    # plt.savefig("simple_path.png") # save as png
    # print("save")
    # plt.show()
    # nx.draw_networkx_nodes(G, pos)
    # nx.draw_networkx_edges(G, pos)
    # nx.draw_networkx_labels(G, pos, labels, font_size=16)

def find(regex, folder='./'):
    found = []
    for filename in glob.iglob(folder+'**/'+regex, recursive=True):
        found.append(filename)
    return found

def read_csvs():    
    org = {}
    csv_files = find("*.csv")

    print("found csv files:", csv_files)
    for csv in csv_files:
        tmp_org = read_org(csv)

        for uid in tmp_org.keys():
            if uid not in org:
                org[uid] = tmp_org[uid]
                missing += 1

def read_org(filename):
    org = {}

    with open(filename, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print('Column names are:', row)
                line_count += 1
            name = row["Name"].split('(')[0]
            parent = row["Reports To"]
            uid = row["Unique Identifier"]
            line_count += 1

            org[uid] = {"name": name, "parent": parent}
        print(f'Processed {line_count} lines.')
    return org

# def read_org(filename):
#     org = []

#     with open(filename, mode='r') as csv_file:
#         csv_reader = csv.DictReader(csv_file)
#         line_count = 0
#         for row in csv_reader:
#             if line_count == 0:
#                 print(f'Column names are {", ".join(row)}')
#                 line_count += 1
#             name = row["Name"].split('(')[0]
#             parent = row["Reports To"]
#             uid = row["Unique Identifier"]
#             line_count += 1

#             org.append({"name": name, "uid": uid, "parent": parent})
#         print(f'Processed {line_count} lines.')
#     return org


if __name__ == "__main__":
    main()
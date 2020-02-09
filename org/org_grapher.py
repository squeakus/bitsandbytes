import networkx as nx
import matplotlib.pyplot as plt
import csv

def main():
    G = nx.DiGraph()
    org = read_org('Bob.txt')
    ann = read_org("AnnKelleher.csv")
    peter = read_org("PeterCharvat.csv")

    for uid in peter.keys():
        if uid not  in ann:
            print(peter[uid]['name'], " is missing")


    # labels = {}
    
    # for idx, emp in enumerate(org):
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
    
    # nx.write_gexf(G, "test.gexf")
    # print("written gxf")
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

def read_org(filename):
    org = {}

    with open(filename, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print('Column names are '.join(row))
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
import networkx as nx
import matplotlib.pyplot as plt
import csv
import pygraphviz
import glob

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
    G = nx.DiGraph()
    #org = read_csvs()
    org = read_org('MurthyRenduchintala.csv')
    
    for name in org.keys():
        G.add_node(name)
        # if name  == "Bob Swan (11637976)":
        #     print(name, org[name]['parent'])
        # elif org[name]['parent'] is '':
        #     G.add_edge("Bob Swan (11637976)", name)
        if org[name]['parent'] is not '':
            G.add_edge(org[name]['parent'], name)
    # output JSON file
    # import json
    # from networkx.readwrite import json_graph
    # data = json_graph.node_link_data(G)
    # with open('org.json', 'w') as fp:
    #     json.dump(data, fp)
    # #json.dump("org.json", data)

    # Plot circular layout
    plt.figure(figsize=(10, 10))
    pos = graphviz_layout(G, prog='twopi', args='')
    nx.draw(G, pos, node_size=5, alpha=0.5, node_color="blue", with_labels=False)
    plt.axis('equal')
    plt.show()

    # use digraph for computing number of employees
    nodes = list(G.nodes)
    print("total", len(nodes))
    # get_level_employees(G, nodes)
    get_employees(G, nodes)

    # depths_graph = nx.shortest_path_length(G, nodes[0])
    # print(depths_graph)
    # levels = list(depths_graph.values())
    # print("Total nodes:", len(nodes))
    # print(nodes[0])
    # print("0:",levels.count(0))
    # print("1:",levels.count(1))
    # print("2:",levels.count(2))
    # print("3:",levels.count(3))
    # print("4:",levels.count(4))
    # print("5:",levels.count(5))
    # print("6:",levels.count(6))
    # print("7:",levels.count(7))
    # print("8:",levels.count(8))
    # print("9:",levels.count(9))
    # print("10:",levels.count(10))
    # plt.hist(levels)
    # plt.show()



    #diameter = nx.algorithms.distance_measures.diameter(G)
    #print("diameter", diameter)
    # # max_clique_size = nx.algorithms.clique.graph_clique_number(G)
    # # print(depth, diameter, max_clique_size)
    
    # nx.write_gexf(G, "fullorg.gexf")
    # print("written gxf")
    # # #pos=nx.spring_layout(G)
    # print("spring")
    # nx.draw(G)
    # print("draw")
    # plt.savefig("simple_path.png") # save as png
    # print("save")
    # plt.show()
    # nx.draw_networkx_nodes(G, pos)
    # nx.draw_networkx_edges(G, pos)
    # nx.draw_networkx_labels(G, pos, labels, font_size=16)

def get_level_employees(G, nodes):
    levels = [[],[],[],[],[],[],[],[],[],[]]
    depths_graph = nx.shortest_path_length(G, nodes[0])
    
    for key in depths_graph.keys():
        staff = len(list(G.successors(key)))
        levels[depths_graph[key]].append(staff)

    print(levels)
    for idx, level in enumerate(levels):
        plt.hist(levels[idx])
        plt.title("level " + str(idx))
        plt.show()

def get_employees(G, nodes):
    reports = []
    singles = 0
    for node in nodes:
        staff = len(list(G.successors(node)))
        if staff > 0:
            reports.append(staff)
        else:
            singles += 1

        if staff > 30:
            print(node, " has ", staff)
    print("singles: ", singles)
    plt.hist(reports, bins=35)
    plt.show()

def get_employees(G, nodes):
    reports = []
    singles = 0
    for node in nodes:
        staff = len(list(G.successors(node)))
        if staff > 0:
            reports.append(staff)
        else:
            singles += 1

        if staff > 30:
            print(node, " has ", staff)
    print("singles: ", singles)
    plt.hist(reports, bins=35)
    plt.show()

def find(regex, folder='./'):
    found = []
    for filename in glob.iglob(folder+'**/'+regex, recursive=True):
        found.append(filename)
    return found

def read_csvs():    
    csv_files = find("*.csv")
    org = read_org('Bob.csv')
    for csv in csv_files:
        print("reading:", csv)
        tmp_org = read_org(csv)

        for name in tmp_org.keys():
            if name not in org:
                org[name] = tmp_org[name]
    return org

def read_org(filename):
    org = {}
    line_count = 0

    with open(filename, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
    
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            name = row["Name"]
            parent = row["Reports To"]
            uid = row["Unique Identifier"]
            u_idx = uid.find('_')
            uid = uid[u_idx+1:]
            p_idx = parent.find('_')
            parent = parent[p_idx+1:]
            line_count += 1

            org[uid] = {"name": name, "parent": parent}
        print(f'Processed {line_count} lines.')
    return org


if __name__ == "__main__":
    main()
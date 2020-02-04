import networkx as nx
import matplotlib.pyplot as plt
import csv

def main():
    G = nx.Graph()
    org = read_org()
    labels = {}
    
    for idx, emp in enumerate(org):
        print("processing " + str(idx))
        G.add_node(emp['uid'])
        labels['uid'] = emp['name']
        if emp['parent'] == '':
            print("itsa null!")
        else:
            G.add_edge(emp['uid'], emp['parent'])

    nx.write_gexf(G, "test.gexf")
    print("written gxf")
    pos=nx.spring_layout(G)
    print("spring")
    nx.draw(G)
    print("draw")
    plt.savefig("simple_path.png") # save as png
    print("save")
    plt.show()
    # nx.draw_networkx_nodes(G, pos)
    # nx.draw_networkx_edges(G, pos)
    # nx.draw_networkx_labels(G, pos, labels, font_size=16)

def read_org():
    org = []

    with open('Bob.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            name = row["Name"].split('(')[0]
            parent = row["Reports To"]
            uid = row["Unique Identifier"]
            print(name + "with UID " + uid + " works for " + parent)
            line_count += 1

            org.append({"name": name, "uid": uid, "parent": parent})
        print(f'Processed {line_count} lines.')
    return org


if __name__ == "__main__":
    main()
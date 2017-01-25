filename = './indiv.1.mesh'
nodes = [[0,0,0]]
edges = []
mesh_file = open(filename,'r')
lines = iter(mesh_file)
for line in lines:
    line = line.rstrip()
    if line == "Vertices":
        counter = int(lines.next())
        for i in range(counter):
            line = lines.next().rstrip()
            array = line.split(' ')
            index = i + 1
            xyz = (int(array[0]), int(array[1]), int(array[2]))
            node = xyz
            nodes.append(node)
    if line == 'Edges':
       counter = int(lines.next())
       for i in range(counter):
           line = lines.next().rstrip()
           array = line.split(' ')
           edge = (int(array[0]), int(array[1]))
           edges.append(edge)
mesh_file.close()
for edge in edges:
    print "connecting", nodes[edge[0]], "and", nodes[edge[1]]


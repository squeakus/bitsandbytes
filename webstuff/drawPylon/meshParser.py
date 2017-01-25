lines_left = True
vert_list, edge_list = [], []
x_max, y_max, z_max = 0,0,0

mesh_file = open('indiv.0.mesh', 'r')

while lines_left:
    line = mesh_file.readline()
    line = line.rstrip()

    if line.startswith("Vertices"):
        vert_count = mesh_file.readline()
        print "no of verts:", vert_count

        for i in range(int(vert_count)):
            vert = mesh_file.readline().rstrip()
            vert = vert.split()
            x, y, z = float(vert[0]), float(vert[1]), float(vert[2])
            sx = x/55000
            sy = y/55000
            sz = z/55000
            vert_dict = {'id':i+1,'x':sx,'y':sy,'z':sz}
            vert_list.append(vert_dict)

    if line.startswith("Edges"):
        edge_count = mesh_file.readline()

        for i in range(int(edge_count)):
            edge = mesh_file.readline().rstrip()
            edge = edge.split()            
            pt_a, pt_b = int(edge[0]), int(edge[1])
            edge_dict = {"pt_a":pt_a,"pt_b":pt_b}
            edge_list.append(edge_dict)
            
    if line.startswith("End"):
        lines_left = False

mesh_file.close()

write_file = open('array.txt','w')

for edge in edge_list:
    for vert in vert_list:
        if vert['id'] == edge['pt_a']:
            xs, ys, zs = str(vert['x']), str(vert['z']), str(vert['y'])
            vert_str =  xs + ", " + ys+", "+zs+","
            print vert_str
            write_file.write(vert_str+'\n')
        if vert['id'] == edge['pt_b']:
            xs, ys, zs = str(vert['x']), str(vert['z']), str(vert['y'])
            vert_str =  xs + ", " + ys+", "+zs+","
            print vert_str
            write_file.write(vert_str+'\n')

write_file.close()
        

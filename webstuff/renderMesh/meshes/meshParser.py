lines_left = True
vert_list, edge_list = [], []
x_max = y_max = z_max = 0
x_min = y_min = z_min = 666

mesh_file = open('test.mesh', 'r')

# read the mesh into a list of vertices
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
            if x > x_max: x_max = x
            if y > y_max: y_max = y
            if z > z_max: z_max = z
	    if x < x_min: x_min = x
	    if y < y_min: y_min = y
	    if z < z_min: z_min = z

            vert_dict = {'id':i+1,'x':x,'y':y,'z':z}
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


#scale the coords to range(0,1)
print "max x", x_max, "min x", x_min
print "max y", y_max, "min y", y_min
print "max z", z_max, "min z", z_min


max_coord = max(x_max, y_max, z_max)
scaled_list = []
for vert in vert_list:
    sx = (vert['x'] - ((x_max + x_min)/2)) / max_coord
    sy = (vert['y'] - ((y_max + y_min)/2)) / max_coord
    sz = (vert['z'] -  z_min) / max_coord
    scaled_dict = {'id':vert['id'],'x':sx,'y':sy,'z':sz}
    scaled_list.append(scaled_dict)

# x_max = y_max = z_max = 0
# x_min = y_min = z_min = 666
# for vert in scaled_list:
#     x, y, z = vert['x'], vert['y'], vert['z']
#     if x > x_max: x_max = x
#     if y > y_max: y_max = y
#     if z > z_max: z_max = z
#     if x < x_min: x_min = x
#     if y < y_min: y_min = y
#     if z < z_min: z_min = z
# print "smax x", x_max, "smin x", x_min
# print "smax y", y_max, "smin y", y_min
# print "smax z", z_max, "smin z", z_min



mesh_file.close()

# generate glmesh for webgl
write_file = open('test.glmesh','w')
for idx, edge in enumerate(edge_list):
    a_x = a_y = a_z = b_x = b_y = b_z = 0
    for vert in scaled_list:
        if vert['id'] == edge['pt_a']:
            a_x, a_y, a_z = str(vert['x']), str(vert['z']), str(vert['y'])
        if vert['id'] == edge['pt_b']:
            b_x, b_y, b_z = str(vert['x']), str(vert['z']), str(vert['y'])

    vert_str =  a_x + ", " + a_y + ", "+ a_z +","
    write_file.write(vert_str+'\n')
    
    # leave comma out for last line
    if idx == len(edge_list)-1:
        vert_str =  b_x + ", " + b_y + ", " + b_z
    else:
        vert_str =  b_x + ", " + b_y + ", " + b_z + ","
        
    write_file.write(vert_str+'\n')



write_file.close()
        

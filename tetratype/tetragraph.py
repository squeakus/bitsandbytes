"""Generates graphs of tetrahedrons"""
import networkx as nx
from math import sqrt, tan
import os, time, ctypes
import analyser

ctypes.cdll.LoadLibrary('libdist.so')
DISTLIB = ctypes.CDLL('libdist.so')

#returns euclidean distance between two points
def distance(pt_1, pt_2):
    """Uses C library to calculate euclidean distance"""
    dist = DISTLIB.distance(pt_1[0], pt_1[1], pt_1[2],
                            pt_2[0], pt_2[1], pt_2[2])
    return dist

def midpoint(pta, ptb):
    """Given two points, returns midpoints """
    x, y, z = (ptb[0]+pta[0])/2, (ptb[1]+pta[1])/2, (ptb[2]+pta[2])/2
    mid = [int(round(x, 0)), int(round(y, 0)), int(round(z, 0))]
    return mid

def mirror(pts, axis, offset = 0):
    """reflects a set of points through an axis"""
    retval = []
    for pt in pts:
        if axis == "x":
            inv1, inv2, inv3 = offset-(pt[0]-offset), pt[1], pt[2]
        if axis == "y":
            inv1, inv2, inv3 = pt[0], offset-(pt[1]-offset), pt[2]
        if axis == "z":
            inv1, inv2, inv3 = pt[0], pt[1], offset-(pt[2]-offset)
        inverse = [inv1, inv2, inv3]
        retval.append(inverse)
    pts.reverse()
    for pt in pts:
        retval.append(pt)
    return retval

def offset_list(pts, axis, offset):
    """move set of points by a given offset"""
    retval = []
    for pt in pts:
        if axis == "x":
            inv1, inv2, inv3 = pt[0]+offset, pt[1], pt[2]
        if axis == "y":
            inv1, inv2, inv3 = pt[0], pt[1]+offset, pt[2]
        if axis == "z":
            inv1, inv2, inv3 = pt[0], pt[1], pt[2]+offset
        inverse = inv1, inv2, inv3
        retval.append(inverse)
    return retval

class Tetragraph(nx.Graph):
    """uses networkx to generate a graph of tetrahedrons"""
    def __init__(self, *args, **kwargs):
        super(Tetragraph, self).__init__(*args, **kwargs)
        self.pop_folder = os.getcwd()+"/population/"
        self.save = True
        self.frame_count = 0
        self.mod = 0

    def get_unused_nodeid(self):
        """The order of a graph is the number of nodes."""
        return self.order()

    def get_max_xyz(self):
        """returns max values, not points"""
        max_x, max_y, max_z = 0, 0, 0
        for node_id in self.nodes():
            x, y, z = self.node[node_id]['xyz']
            if max_x < x: 
                max_x = x
            if max_y < y: 
                max_y = y
            if max_z < z: 
                max_z = z
        return [max_x, max_y, max_z]

    def get_min_xyz(self):
        """returns min values, not points"""
        min_x, min_y, min_z = 5000, 5000, 5000
        for node_id in self.nodes():
            x, y, z = self.node[node_id]['xyz']
            if min_x > x: 
                min_x = x
            if min_y > y: 
                min_y = y
            if min_z > z: 
                min_z = z
        return [min_x, min_y, min_z]

    def nearest_node(self, x, y, z):
        """returns the distance from the nearest node on the graph"""
        pos = [x, y, z]
        dist_sort = lambda k: distance(self.node[k]['xyz'], pos)
        nn = min(self.node, key=dist_sort)
        dist_value = distance(self.node[nn]['xyz'], pos)
        return dist_value

    def add_unique_node(self, coords, node_type):
        """ if it already exists, just return that node ID"""
        new = True
        pta = [int(coords[0]), int(coords[1]), int(coords[2])]
        for node in self.node:
            ptb = self.node[node]['xyz']
            ptb = [int(ptb[0]), int(ptb[1]), int(ptb[2])]
            if pta == ptb:
                new = False
                node_id = node
                break
        if new:
            node_id = self.get_unused_nodeid()
            self.add_node(node_id, xyz= pta, label = node_type)
        return node_id

    def connect_neighbours(self, node_list, length):
        """connect all nodes within a given distance"""
        tolerance = 5
        for a_node in node_list:
            for b_node in node_list:
                if not self.has_edge(a_node, b_node):
                    pt1 = self.node[a_node]["xyz"]
                    pt2 = self.node[b_node]["xyz"]
                    #print "distance",distance(pt1,pt2)
                    if abs(int(distance(pt1, pt2))-length) < tolerance:
                        self.add_edge(a_node, b_node)
                        #self.save_graph()

    def create_mesh(self, name):
        """generates a .mesh file from the graph"""
        node_list = []
        edge_list = []
        for node_id in self.nodes():
            node = {}
            xyz = self.node[node_id]['xyz']
            x, y, z = xyz[0], xyz[1], xyz[2]
            label = self.node[node_id]['label']
            node = {'id':str(node), 'x':x, 'y':y, 'z':z, 'label':label}
            node_list.append(node)
        for idx, edge in enumerate(self.edges_iter()):
            edge = {'id':idx, 'pta':edge[0], 'ptb':edge[1]}
            edge_list.append(edge)   
        
        filename = name+'.mesh' 
        mesh = open(filename, 'w')
        mesh.write("MeshVersionFormatted 1\nDimension\n3 \n")
        mesh.write("Vertices\n"+str(len(node_list))+" \n")
        for node in node_list:
            mesh.write(str(node['x'])+" "+str(node['y'])+
                       " "+str(node['z'])+" 0  \n")
        mesh.write("Edges\n"+str(len(edge_list))+" \n")
        for edge in edge_list:
            pta, ptb = int(edge['pta']), int(edge['ptb'])
            mesh.write(str(pta+1)+" "+str(ptb+1)+" 0 \n")
        mesh.write("End\n")
        mesh.close()

    def save_graph(self, count = None):
        if self.save:
            if count == None:
                if self.frame_count % 50 == 0:
                    self.mod += 1
                filename = self.pop_folder+"img%04d"% (self.frame_count)
                self.frame_count += 1
                if self.frame_count % self.mod == 0:
                    self.create_mesh(filename)
            else:
                filename = self.pop_folder+"gen%04d"% (count)
                self.create_mesh(filename)
                
    def regular(self, origin, length):
        """creates a regular tetrahedron(4 equilateral triangles)"""
        #use the first coordinate to define the rest
        height = int(origin[2]+(sqrt(6)/3)*length)
        x0, y0, z0 = int(origin[0]), int(origin[1]), int(origin[2])
        pt0 = [x0, y0, z0]
        x1, y1, z1 = int(x0+length), int(y0), int(z0)
        pt1 = [x1, y1, z1]
        x2, y2, z2 = int(x0+length/2), int(y0+sqrt(length**2-((length/2)**2))), z0
        pt2 = [x2, y2, z2]
        #don't forget to use rads
        x3, y3, z3 = int(x2), int(y0+tan(0.5235)*(length/2)), height 
        pt3 = [x3, y3, z3]
        node_list = [pt0, pt1, pt2, pt3]
        #add pts as nodes to a graph
        id_list = []
        for node in node_list:
            if node[2] == 0:
                id_list.append(self.add_unique_node(node,"fixed")) 
            else:
                id_list.append(self.add_unique_node(node,"none"))  
        self.connect_neighbours(id_list, length)
        return node_list

    def square(self, origin, length):
        """creates a square tetrahedron(3 triangles,square base)"""
        #define points starting from origin
        height = int((sqrt(2)*length)/2)
        x0, y0, z0 = int(origin[0]), int(origin[1]), int(origin[2])
        pt0 = [x0, y0, z0]
        x1, y1, z1 = int(x0+length), int(y0), int(z0)
        pt1 = [x1, y1, z1]
        x2, y2, z2 = int(x0+length), int(y0+length), int(z0)
        pt2 = [x2, y2, z2]
        x3, y3, z3 = int(x0), int(y0+length), int(z0)
        pt3 = [x3, y3, z3]
        mid = midpoint(pt0, pt2)
        x4, y4, z4 = int(mid[0]), int(mid[1]), int(z0+height)
        pt4 = [x4, y4, z4]
        # add points to the graph
        node_list = [pt0, pt1, pt2, pt3, pt4]
        id_list = []
        for node in node_list:
            if node[2] == 0:
                id_list.append(self.add_unique_node(node,"fixed")) 
            else:
                id_list.append(self.add_unique_node(node,"none"))     
        self.connect_neighbours(id_list, length)
        return node_list

    def octahedron(self, origin, length):
        """uses square and mirror methods to create octahedron"""
        id_list = []
        tmp = Tetragraph()
        nodes = tmp.square(origin, length)
        points = mirror(nodes, 'z', origin[2])
        for node in points:
            if node[2] == 0:
                id_list.append(self.add_unique_node(node,"fixed")) 
            else:
                id_list.append(self.add_unique_node(node,"load"))
        self.connect_neighbours(id_list, length)
        points = points[2:] 
        return points

    def recurse(self, func, origin, length, depth):
        """calls a given method recursively to a given depth"""
        next_nodes = [] 
        if depth == 1:
            start_nodes = func(origin, length)        
            next_nodes.append(origin)
            for i in range(1, len(start_nodes)):
                tmp_nodes = func(start_nodes[i], length)
                next_nodes.append(tmp_nodes[i])
            return next_nodes 
        else:
            start_nodes = self.recurse(func, origin, length, depth-1)
            next_nodes.append(start_nodes[0])
            for i in range(1, len(start_nodes)):
                tmp_nodes = self.recurse(func, start_nodes[i], length, depth-1)
                next_nodes.append(tmp_nodes[i])
            return next_nodes

    def grid(self, func, origin, length, size):
        """generates 3D cube of a given function"""
        current = origin
        floor_nodes = []
        for x in range(size):
            for y in range(size):
                func(current, length)
                current = [current[0], current[1]+length, origin[2]]
            current = [current[0]+length, origin[1], current[2]]

        for node_id in self.nodes():
            floor_nodes.append(self.node[node_id]['xyz'])
            
        z_offset = int(sqrt(2)*length)
        for z in range(size/2):
            floor_nodes = offset_list(floor_nodes, 'z', z_offset)
            id_list = []
            for node in floor_nodes:
                id_list.append(self.add_unique_node(node, "load"))  
            self.connect_neighbours(id_list, length) 

    def sanitize(self):
        """removes disconnected nodes and reorders node ids"""
        lonely_nodes = [n for n, d in self.degree_iter() if d == 0]
        self.remove_nodes_from(lonely_nodes)
        self = nx.connected_component_subgraphs(self)[0]
        self = nx.convert_node_labels_to_integers(self) 
        return self

    def generate_graph(self, genome):
        """uses chromosome to generate graph"""
        node_count = self.number_of_nodes()
        for i in range(node_count):
            if genome[i] < 1:
                self.remove_node(i)
        self = self.sanitize()
        return self

if __name__ == "__main__":
    TIME = time.time()
    LENGTH = 1000 #make it big to keep it in integers
    DEPTH = 3
    HEIGHT = int((sqrt(2)*LENGTH)/2)
    TETRA = Tetragraph()
    TETRA.grid(TETRA.octahedron, [0, 0, HEIGHT], LENGTH, 10)  
#    TETRA.recurse(TETRA.regular, [0,0,0], LENGTH, DEPTH)
#    TETRA.recurse(TETRA.regular,[0,0,0],LENGTH,DEPTH)
    OCT_HEIGHT = HEIGHT*(2**DEPTH)
    OCT_START = [0, 0, OCT_HEIGHT]
    TETRA.recurse(TETRA.octahedron, OCT_START, LENGTH, DEPTH)
    
    TETRA.save = True
    TETRA.save_graph(9999)
    print "max xyz", str(TETRA.get_max_xyz())
    print "min xyz", str(TETRA.get_min_xyz())
    print "nodes:", TETRA.number_of_nodes(), "edges", TETRA.number_of_edges()
    print "total time: %.2f" % (time.time() - TIME)
    print "final frame: ", str(TETRA.frame_count-1)

    #test chromosome
    #genome = [random.choice([0,0,0,0,1]) for i in xrange(node_count)]
    #TETRA = TETRA.generate_graph(genome)
    #TETRA.save_graph(998)

    #will it blend?
    AZR = analyser.Analyser('test', "moo", True)
    AZR.myGraph = TETRA
    AZR.parse_graph(TETRA)
    AZR.apply_stresses()
    AZR.create_slf_file()
    AZR.test_slf_file()
    AZR.parse_results()
    AZR.print_stresses()
    AZR.show_analysis()

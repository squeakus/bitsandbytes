import networkx as nx
import ctypes
from geometry import *

# Using C library to compute distance
ctypes.cdll.LoadLibrary('libdist.so')
DISTLIB = ctypes.CDLL('libdist.so')

#returns euclidean distance between two points
def distance(pt_1, pt_2):
    """Uses C library to calculate euclidean distance"""
    dist = DISTLIB.distance(int(pt_1[0]), int(pt_1[1]), int(pt_1[2]),
                            int(pt_2[0]), int(pt_2[1]), int(pt_2[2]))
    return dist

# This class stores the current position, orientation, and node.
# Idea is: there's a "cursor", like a turtle, which moves around,
# rotates, etc. There's also currentNode which isn't used for now.
# As in L-systems, we have save and restore functions for state, 
# which is stored in a stack.
class GraphState(object):
    def __init__(self, pos=None, orientation=None):
        if pos is None:
            # x, y, z
            self.position = 0, 0, 0
        else:
            self.position = pos
        if orientation is None:
            # orientation is a vector
            self.orientation = 1, 0, 0
        else:
            self.orientation = orientation
        self.currentNode = 0

## A class used for representing an entire blender wood-beam
## scene. Nodes and edges.
class graph(nx.Graph):
    def __init__(self, *args, **kwargs):
        super(graph, self).__init__(*args, **kwargs)
        self.states = [GraphState()]
        self.state = self.states[0]

    # Call this first, to make sure that every graph is non-empty.
    def initial_edge(self, d):
        self.add_node(0, xyz=self.state.position)
        self.project(d)
        self.add_node(1, xyz=self.state.position)
        self.add_edge(0, 1)

    # The order of a graph is the number of nodes. Nodes are
    # numbered sequentially as we add them (see add_edge), so the
    # order is the number of the next free id.
    def get_unused_nodeid(self):
        return self.order()

    # save and restore methods for the stack of states.
    def saveState(self):
        self.states.append(copy.copy(self.state))

    def restoreState(self):
        self.state = self.states.pop()
        if self.state is None:
            self.state = GraphState()

    # "Rotate" the turtle's orientation. Not a true rotation --
    # just addition.
    def rotate(self, a, b, c):
        self.state.orientation = (self.state.orientation[0] + a,
                                  self.state.orientation[1] + b,
                                  self.state.orientation[2] + c)

    # Move the turtle, don't add any edge.
    def move(self, x, y, z):
        self.state.position = (self.state.position[0] + x,
                               self.state.position[1] + y,
                               self.state.position[2] + z)

    # Move the turtle, taking account of orientation. Don't add
    # any edge.
    def project(self, d):
        self.move(d * self.state.orientation[0],
                  d * self.state.orientation[1],
                  d * self.state.orientation[2])

    # What node id is nearest turtle's current position? Unused in
    # blender_graph.bnf for now.
    def nearestNodeId(self):
        pos = self.state.position
        dist = lambda k: euclidean_distance(self.node[k]['xyz'], pos)
        return min(self.node, key=dist)

    def nearest_node(self, x, y, z):
        """returns the distance from the nearest node on the graph"""
        pos = [x, y, z]
        dist_sort = lambda k: distance(self.node[k]['xyz'], pos)
        nn = min(self.node, key=dist_sort)
        dist_value = distance(self.node[nn]['xyz'], pos)
        return dist_value

    # New currentNode.
    def incrCurrentNode(self):
        self.state.currentNode += 1
        self.state.currentNode %= self.order()

    # Add a node and connect it to two others, ensuring that it's
    # not lonesome.
    def add_unique_node(self,XYZ, nodeType):
        new = True
        x,y,z = XYZ[0],XYZ[1],XYZ[2]
        AXYZ = [round(x,3),round(y,3),round(z,3)]
        for node in self.node:
            XYZ = self.node[node]['xyz']
            x,y,z = XYZ[0],XYZ[1],XYZ[2]
            BXYZ = [round(x,3),round(y,3),round(z,3)]
            #print "a:",AXYZ,"b",BXYZ
            if AXYZ == BXYZ:
                new = False
                id = node
                break
        if new:
            id = self.get_unused_nodeid()
            self.add_node(id, xyz= AXYZ, label = nodeType)
        return id

    def add_node_connect2(self, id1, id2):
        nodeid = self.get_unused_nodeid()
        self.add_node(nodeid, xyz=self.state.position)
        self.add_edge(nodeid, id1)
        self.add_edge(nodeid, id2)

    def connect_nodes(self, nodeList):
        nodeList.sort()
        for i in range(len(nodeList) - 1):
            self.add_edge(nodeList[i], nodeList[i+1])

    # Given a node id, and a radius, add a number of new nodes and
    # edges from each to the central node.
    def add_star(self, node, radius, npts):
        node = self.get_node_idx_mod(node)
        xyz = self.get_node_data(node)
        for i in range(npts):
            theta = 2 * pi * i / npts
            newxyz = xyz[0] + cos(theta), xyz[1] + 3, xyz[2] + sin(theta)
            new_nodeid = self.get_unused_nodeid()
            self.add_node(new_nodeid, None, xyz=newxyz)
            self.add_edge(node, new_nodeid)

    # Assume x and y exist.
    def add_edge_between_existing_nodes(self, x, y):
        self.add_edge(x, y)

    # Given an integer, return a node id guaranteed to exist.
    def get_node_idx_mod(self, x):
        return x % self.order()

    # Given an integer x, return a node id guaranteed to exist,
    # and NOT node id y.
    def get_node_idx_mod_exclude_y(self, x, y):
        tmp = x % (self.order() - 1)
        if tmp >= y:
            tmp += 1
        return tmp

    # Given a number, return a node id guaranteed to exist.
    def get_node_idx_float(self, x):
        return int(x * self.order())

    # Given a number, return a node id guaranteed to exist, not y.
    def get_node_idx_float_exclude_y(self, x, y):
        tmp = int(x * (self.order() - 1))
        if tmp >= y:
            tmp += 1
        return tmp

    # What nodes have exactly n edges?
    def get_nodes_with_n_edges(self, n):
        return [node for node in self.adj
                if self.degree(node) == n]

    # Return a particular node of those with m edges.
    def get_nth_node_with_m_edges_mod(self, n, m):
        nodes = self.get_nodes_with_n_edges(m)
        if len(nodes):
            return nodes[n % len(nodes)]
        else:
            return None

    # Save a 2d-visualisation of the graph (ignoring 3d attributes of the 
    # blender scene). Requires matplotlib.
    def save_picture(self, filename):
        nx.draw_graphviz(self)

    def check_graph(self):
        dupCnt = 0
        for nodeA in self.node:
            for nodeB in self.node:
                if nodeA != nodeB:
                    Axyz = self.node[nodeA]["xyz"]
                    Bxyz = self.node[nodeB]["xyz"]
                    if Axyz == Bxyz:
                        print "A:",self.node[nodeA]["xyz"],"B:",self.node[nodeA]["xyz"]
                        dupCnt += 1
        if dupCnt > 0:
            print "duplicates!",dupCnt
        else:
            print "no duplicates"

    # Pop up a 2d-visualisation of the graph (ignoring 3d
    # attributes of the blender scene). Requires matplotlib.
    def show_picture(self):
        nx.draw_graphviz(self)
        #plt.show()

    # Get the 3d attributes of a node.
    def get_node_data(self, nodeid):
        return self.node[nodeid]["xyz"]

    # Get the 3d attributes of a node.
    def get_xyz(self, nodeid):
        x,y,z = self.node[nodeid]["xyz"][0],self.node[nodeid]["xyz"][1],self.node[nodeid]["xyz"][2]
        return x,y,z
    # Get the 3d attributes of a node.
    def get_label(self, nodeid):
        return self.node[nodeid]["label"]

    # Add a copy of the graph, offsetting all nodes by a given
    # vector. For nodes with the "rung" attribute, add an edge
    # between existing node and its offset copy.
    def copy_and_offset_with_mirror(self, offset, mirror=False):
        # make an unchanged copy and an offset/mirrored copy
        orig_copy = self.copy()
        offset_copy = self.copy()
        for nodeid in offset_copy.node:
            # perform an offset
            xyz = offset_copy.node[nodeid]["xyz"]
            xyz = pt_plus_pt(xyz, offset)
            if mirror:
                ## also perform a mirror in the y axis
                xyz = [xyz[0], -xyz[1], xyz[2]]
            offset_copy.node[nodeid]["xyz"] = xyz

        # make a union of the original and copy, renaming nodes
        # note that this requires nx to be updated to svn 1520 or above
        # which fixes a bug where union discards node attributes
        new_graph = nx.union(orig_copy, offset_copy, rename=("G-", "H-"))

        # make edges between corresponding nodes in original and copy where needed
        duplicates = []
        for nodeid in new_graph.node:
            if nodeid.startswith("G-"):
                #check for same coords
                h_node_id = nodeid.replace("G", "H")
                Gxyz = new_graph.node[nodeid]['xyz']
                Hxyz = new_graph.node[h_node_id]['xyz']
                if Hxyz == Gxyz:
                    duplicates.append([nodeid,h_node_id])
                #connect walkways
                if  new_graph.node[nodeid]['label']=='walkway': 
                    new_graph.node[h_node_id]['label']= 'walkway'
                    h_node = new_graph.node[h_node_id]
                    new_graph.add_edge(nodeid, h_node_id,label='walkway')
                if  new_graph.node[nodeid]['label']=='join': 
                    new_graph.node[h_node_id]['label']= 'join'
                    h_node = new_graph.node[h_node_id]
                    new_graph.add_edge(nodeid, h_node_id,label='join')

        for duplicate in duplicates:
            for edge in new_graph.edges_iter(data=True):
                if edge[0] == duplicate[1]:
                    new_graph.add_edge(edge[1],duplicate[0])
                elif edge[1] == duplicate[1]:
                    new_graph.add_edge(edge[0],duplicate[0])                   
            new_graph.remove_node(duplicate[1])
        
        new_graph = nx.convert_node_labels_to_integers(new_graph)
        self.clear()

        #only add nodes that are connected to edges
        for edge in new_graph.edges_iter(data=True):
            if edge[0]==edge[1]:
                new_graph.remove_edge(edge[0],edge[0])
            else:
                self.add_node(edge[0], xyz=new_graph.node[edge[0]]["xyz"],label=new_graph.node[edge[0]]['label'])
                self.add_node(edge[1], xyz=new_graph.node[edge[1]]["xyz"],label=new_graph.node[edge[1]]['label'])
                self.add_edge(edge[0],edge[1],edge[2])


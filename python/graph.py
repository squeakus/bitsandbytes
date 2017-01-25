graphSize = 200

def euclidean_distance(p, q):
    return sqrt(sum([(p[i] - q[i]) **2 for i in range(len(p))]))

from math import pi,cos,sin,sqrt
import os
import random
import matplotlib.pyplot as plt
import sys
import networkx as nx

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

    # prints out the degree of each node
    def connectedness(self):
        for node in self.adj:
            print "node:",str(node)," has degree: ",str(self.degree(node))

    # New currentNode.
    def incrCurrentNode(self):
        self.state.currentNode += 1
        self.state.currentNode %= self.order()

    # Add a node and connect it to a given node, ensuring that it's
    # not lonesome.
    def add_node_connect(self, id1):
        nodeid = self.get_unused_nodeid()
        self.add_node(nodeid, xyz=self.state.position)
        self.add_edge(nodeid, id1)

    # Add a node and connect it to two others, ensuring that it's
    # not lonesome.
    def add_node_connect2(self, id1, id2):
        nodeid = self.get_unused_nodeid()
        self.add_node(nodeid, xyz=self.state.position)
        self.add_edge(nodeid, id1)
        self.add_edge(nodeid, id2)

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

    def interpolate(p, xy):
        p = 1 - p
        x, y = xy
        x0, y0, z0 = x
        x1, y1, z1 = y
        return [x0 * p + x1 * (1 - p), y0 * p + y1 * (1 - p), z0 * p + z1 * (1 - p)]

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
        plt.clf()
        nx.draw_graphviz(self)
        plt.savefig(filename)

    # Pop up a 2d-visualisation of the graph (ignoring 3d
    # attributes of the blender scene). Requires matplotlib.
    def show_picture(self):
        nx.draw_graphviz(self)
        plt.show()

    # Get the 3d attributes of a node.
    def get_node_data(self, nodeid):
        return self.node[nodeid]["xyz"]

    # Convert our graph into a list of blender objects.
    def get_blender_objs(self):
        retval = []
        for edge in self.edges_iter():
            x, y = (self.get_node_data(edge[0]), self.get_node_data(edge[1]))
            retval.append(render.connect(x, y))
        return retval

    # draw <tournament_size> nodes from the graph and pick the 
    # node with the highest degree
    def tournament_select(self, tournament_size=3):
        winner = 0
        if self.order() <= tournament_size:
            winner = self.get_node_idx_mod(random.randint(0,graphSize))
            #winner = g.get_unused_nodeid()             
        else:
            competitors = random.sample(self.adj, tournament_size)
            degree = lambda k: self.degree(k)
            winner = max(competitors,key = degree)
        return winner

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
        for nodeid in new_graph.node:
            if nodeid.startswith("G-"):
                # print "looking at", nodeid
                if "rung" in new_graph.node[nodeid]:
                    h_node_id = nodeid.replace("G", "H")
                    h_node = new_graph.node[h_node_id]
                    new_graph.add_edge(nodeid, h_node_id)
                    # print("making edge between " + 
                    #       str(new_graph.node[nodeid]["xyz"]) + 
                    #       " : " + 
                    #       str(new_graph.node[h_node_id]["xyz"]))


        # rename nodes back to integers: FIXME doesn't work because bug in nx
        # strips node attributes.
        # new_graph = nx.convert_node_labels_to_integers(new_graph)

        # clear self and add edges from new graph
        self.clear()
        for edge in new_graph.edges_iter():
            self.add_node(edge[0], xyz=new_graph.node[edge[0]]["xyz"])
            self.add_node(edge[1], xyz=new_graph.node[edge[1]]["xyz"])
            self.add_edge(edge[0], edge[1])




if __name__ == "__main__":
    g = graph()
    g.initial_edge(5)

    for i in range(0,graphSize):
        g.add_node_connect(g.tournament_select(5))
                
        #filename = "%s%03d" % ("img",i)
        #fullFilename  = os.getcwd()+"/images/"+filename
        #print fullFilename
        #g.save_picture(fullFilename)

        #g.add_node_connect2(g.get_unused_nodeid(),g.get_node_idx_mod(random.randint(0,graphSize)))
        # if random.choice([True,False,False]):
        #     g.add_node_connect(1) 
        # else:
        #     g.add_node_connect2(i,i+1)
        #
    #g.add_star(1,3,4)
    #g.add_edge_between_existing_nodes(0,4)
    #g.add_star(4,3,2)
    #g.add_edge_between_existing_nodes(2,7)
    #g.connectedness()
    g.show_picture()
    print "nearest node: ",g.nearestNodeId()
    print "created graph and added edge"
    


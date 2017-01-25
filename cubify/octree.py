from vectors import pnt2line, distance

def main():
    rootcube = [-2.0, -2.0, -2.0, 2.0, 2.0, 2.0]
    points = [(0,0,0),(0,1,1), (0,-1.5,-1.5),(1.8,-1.5,-1.5)]
    rootnode = Node(None, rootcube)
    tree = Octree(rootnode, rootcube, points)

class Node():
    ROOT, BRANCH, LEAF = 0, 1, 2
    maxdepth = 7
    #1-3 green 4-6 red
    colors = [(0, 100, 0),(0, 150, 0),(0, 200, 0),(0, 255, 0),
              (100, 0, 0),(150, 0, 0),(200, 0, 0),(255, 0, 0)]

    leafcolors = [(255,0,0),(0,255,0),(255,0,255),(255,255,0),
                  (0,0,255),(0,0,255),(0,0,255),(0,0,255),(0,0,255)]

    def __init__(self, parent, cube):
        self.parent = parent
        self.cube = cube
        self.subcubes = []
        self.children = [None] * 8
        
        if parent == None:
            self.type = Node.ROOT
            self.depth = 0
        else:
            self.depth = parent.depth + 1

        #self.color = Node.colors[self.depth]
        self.color = (255,0,0)
        if self.depth >= Node.maxdepth:
            self.type = Node.LEAF
        else:
            self.type = Node.BRANCH

    # Recursively subdivides a cubeangle. Division occurs 
    # ONLY if the cubeangle spans a "feature of interest".
    def addpoint(self, point):
        if self.type == Node.LEAF:
            return

        if len(self.subcubes) == 0:
            self.subdivide()
        
        for idx, cube in enumerate(self.subcubes):
            span = self.spans_point(cube, point)
            
            if span == True:
                # if it doesn't already exist
                if self.children[idx] == None:
                    self.children[idx] = Node(self,self.subcubes[idx])
                self.children[idx].addpoint(point) # << recursion
                break

    def subdivide(self):
        x0,y0,z0,x1,y1,z1 = self.cube
        h = (x1 - x0)/2
        w = (y1 - y0)/2
        d = (z1 - z0)/2
        self.subcubes.append((x0, y0, z0, x0 + h, y0 + w, z0 + d))
        self.subcubes.append((x0, y0 + w, z0, x0 + h, y1, z0 + d))
        self.subcubes.append((x0 + h, y0, z0, x1, y0 + w, z0 + d))
        self.subcubes.append((x0 + h, y0 + w, z0, x1, y1, z0 + d))
        self.subcubes.append((x0, y0, z0 + d, x0 + h, y0 + w, z1))
        self.subcubes.append((x0, y0 + w, z0 + d, x0 + h, y1, z1))
        self.subcubes.append((x0 + h, y0, z0 + d, x1, y0 + w, z1))
        self.subcubes.append((x0 + h, y0 + w, z0 + d, x1, y1, z1))

    def getedges(self):
        x0, y0, z0, x1, y1, z1 = self.cube
        edges = ( ((x0,y0,z0),(x1,y0,z0)),
                  ((x1,y0,z0),(x1,y1,z0)), 
                  ((x1,y1,z0),(x0,y1,z0)), 
                  ((x0,y1,z0),(x0,y0,z0)),
                  ((x0,y0,z1),(x1,y0,z1)),
                  ((x1,y0,z1),(x1,y1,z1)), 
                  ((x1,y1,z1),(x0,y1,z1)), 
                  ((x0,y1,z1),(x0,y0,z1)),
                  ((x0,y0,z0),(x0,y0,z1)),
                  ((x1,y0,z0),(x1,y0,z1)), 
                  ((x1,y1,z0),(x1,y1,z1)), 
                  ((x0,y1,z0),(x0,y1,z1))) 
        return edges
        
    def getvertices(self):
        x0, y0, z0, x1, y1, z1 = self.cube
        verts = ((x0, y0, z0),
                 (x1, y0, z0),
                 (x1, y1, z0),
                 (x0, y1, z0),
                 (x0, y0, z1),
                 (x1, y0, z1),
                 (x1, y1, z1),
                 (x0, y1, z1))
        return verts

    def getcenter(self):
        x0, y0, z0, x1, y1, z1 = self.cube
        x = x0 + ((x1-x0)/2)
        y = y0 + ((y1-y0)/2)
        z = z0 + ((z1-z0)/2)
        return (x,y,z)

    def getlengths(self):
        x0, y0, z0, x1, y1, z1 = self.cube
        x = x1 - x0
        y = y1 - y0
        z = z1 - z0
        if z < 0 or y < 0 or x < 0:
            print "NEGATIVE POINT BAD, min should be first"
        return (x,y,z)

    def spans_point(self, cube, point):
        x0,y0,z0,x1,y1,z1 = cube
        x,y,z = point[0],point[1],point[2]
        if x0 <= x < x1:
            if y0 <= y < y1:
                if z0 <= z < z1:
                    return True                    
        return False
            
class Octree():
    maxdepth = 1 # the "depth" of the tree
    leaves = []
    branches = []
    allnodes = []

    def __init__(self, rootnode, mincube, points):
        Node.minsize = mincube
        Octree.points = points
        Octree.rootnode = rootnode
        for idx, point in enumerate(Octree.points):
            Octree.rootnode.addpoint(point) # constructs the network of nodes
        #self.z_filter(Octree.rootnode)
        self.prune(Octree.rootnode)
        self.traverse(Octree.rootnode)
        print "Points:",len(points),"Nodes:", len(Octree.allnodes), "Branches:",len(Octree.branches), "Leaves:", len(Octree.leaves), "maxdepth:", Octree.maxdepth

    def z_filter(self, node):
        if node.type == Node.LEAF:
            minz = node.cube[2]
            if minz < 5:
                return False
        else:
            for i in range(len(node.children)):
                if node.children[i] != None:
                    if not self.z_filter(node.children[i]):
                        node.children[i] = None
        return True
        
    # Sets children of 'node' to None if they do not have any
    # LEAF nodes.        
    def prune(self, node):
        if node.type == Node.LEAF:
            return 1
        leafcount = 0
        removals = []
        childcnt = 0
        for child in node.children:
            if child != None:
                childcnt += 1
                leafcount += self.prune(child)

        # if leafcount <= 1:
        #     for idx, child in enumerate(node.children):
        #         if not child == None:
        #             child.color =(0,0,255)
        #             #node.children[idx] = None
        # if childcnt == 8:
        #     node.color = (0,255,0)
            
        # color depending on number of neighbours.
        # for child in node.children:
        #     #print child
        #     if child != None:
        #         #print childcnt-1, leafcount
        #         child.color = Node.leafcolors[childcnt-1]


        return leafcount
    # Appends all nodes to a "generic" list, but only LEAF 
    # nodes are appended to the list of leaves.
    def traverse(self, node):
        Octree.allnodes.append(node)
        if node.type == Node.BRANCH:
            Octree.branches.append(node)
        if node.type == Node.LEAF:
            Octree.leaves.append(node)
            if node.depth > Octree.maxdepth:
                Octree.maxdepth = node.depth
        for child in node.children:
            if child != None:
                self.traverse(child) # << recursion

    def create_graph(self):
        import networkx as nx
        G=nx.balanced_tree(4,2)
        pos = nx.graphviz_layout(G,prog='twopi',args='')
        nx.draw(G,pos)

if __name__=='__main__':
    main()

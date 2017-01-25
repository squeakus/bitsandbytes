from vectors import pnt2line, distance

class Node():
    ROOT, BRANCH, LEAF = 0, 1, 2
    minsize = 1

    def __init__(self, parent, cube):
        self.parent = parent
        self.cube = cube
        x0,y0,z0,x1,y1,z1 = cube
        self.children = [None] * 8
        
        if parent == None:
            self.type = Node.ROOT
            self.depth = 0
        else:
            self.depth = parent.depth + 1

        if (x1 - x0) <= Node.minsize:
            self.type = Node.LEAF
        else:
            self.type = Node.BRANCH


    # Recursively subdivides a cubeangle. Division occurs 
    # ONLY if the cubeangle spans a "feature of interest".
    def subdivide(self):
        if self.type == Node.LEAF:
            return

        x0,y0,z0,x1,y1,z1 = self.cube
        h = (x1 - x0)/2
        w = (y1 - y0)/2
        d = (z1 - z0)/2
        cubes = []
        cubes.append((x0, y0, z0, x0 + h, y0 + w, z0 + d))
        cubes.append((x0, y0 + w, z0, x0 + h, y1, z0 + d))
        cubes.append((x0 + h, y0, z0, x1, y0 + w, z0 + d))
        cubes.append((x0 + h, y0 + w, z0, x1, y1, z0 + d))
        cubes.append((x0, y0, z0 + d, x0 + h, y0 + w, z1))
        cubes.append((x0, y0 + w, z0 + d, x0 + h, y1, z1))
        cubes.append((x0 + h, y0, z0 + d, x1, y0 + w, z1))
        cubes.append((x0 + h, y0 + w, z0 + d, x1, y1, z1))
        
        for n in range(len(cubes)):
            span = self.spans_points(cubes[n])
            
            if span == True:
                self.children[n] = self.getinstance(cubes[n])
                self.children[n].subdivide() # << recursion
                
    # A utility proc that returns True if the coordinates of
    # a point are within the bounding box of the node.
    def contains(self, x, y, z):
        x0,y0,z0,x1,y1,z1 = self.cube
        if x >= x0 and x <= x1:
            if  y >= y0 and y <= y1:
                if z >= z0 and z <= z1:
                    return True
        return False
        
    # Sub-classes must override these two methods.
    def getinstance(self,cube):
        return Node(self,cube)

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

    def spans_points(self, cube):
        x0,y0,z0,x1,y1,z1 = cube
        
        if self.depth < 6: # this may require adjustment
            for point in Octree.points:
                x,y,z = point[0],point[1],point[2]
                if x0 < x < x1:
                    if y0 < y < y1:
                        if z0 < z < z1:
                            return True                    
            return False

    def spans_feature(self, cube):
        x0,y0,z0,x1,y1,z1 = cube
        if y0 < 0 or y1 < 0:
            return False
        if self.depth < 6: # this may require adjustment
            for circle in Octree.circles:
                rad,x,y,z = circle
                edges = self.getedges()
                for edge in edges:
                    dist,loc = pnt2line((x,y,z), edge[0], edge[1])
                    if dist <= rad:
                        return True
        verts = [(x0,y0,z0),(x0,y1,z0),(x1,y1,z0),(x1,y0,z0),
                 (x0,y0,z1),(x0,y1,z1),(x1,y1,z1),(x1,y0,z1)]

        for circle in Octree.circles:
            rad,x,y,z = circle
            center = (x,y,z)
            span = 0
            
            for vert in verts:
                d = distance(vert,center)
                span += (d <= rad)
            #if span > 0: print span
            if span > 0 and span < 8:
                return True
        return False

class Octree():
    maxdepth = 1 # the "depth" of the tree
    leaves = []
    branches = []
    allnodes = []
    
    def __init__(self, rootnode, mincube, circles, points):
        Node.minsize = mincube
        Octree.circles = circles
        Octree.points = points
        rootnode.subdivide() # constructs the network of nodes
        self.prune(rootnode)
        self.traverse(rootnode)

    # Sets children of 'node' to None if they do not have any
    # LEAF nodes.        
    def prune(self, node):
        if node.type == Node.LEAF:
            return 1
        leafcount = 0
        removals = []
        for child in node.children:
            if child != None:
                leafcount += self.prune(child)
                if leafcount == 0:
                    removals.append(child)
        for item in removals:
            n = node.children.index(item)
            node.children[n] = None        
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

def main():
    circles = [(1.9,0,0,0),(1.0,0.95,0,0)]
    rootcube = [-2.0, -2.0, -2.0, 2.0, 2.0, 2.0]
    resolution = 0.02
    
    rootnode = Node(None, rootcube)
    tree = Octree(rootnode, resolution, circles)
    print('Wrote %d polygons' % len(tree.leaves))



if __name__=='__main__':
    main()

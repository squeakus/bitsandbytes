# quadtree implementation by Malcolm Kesson

class Node():
    ROOT = 0
    BRANCH = 1
    LEAF = 2
    minsize = 1
    
    def __init__(self, parent, rect):
        self.parent = parent
        self.children = [None,None,None,None]

        if parent == None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1
            
        self.rect = rect
        x0,y0,x1,y1 = rect
        
        if self.parent == None:
            self.type = Node.ROOT
        elif (x1 - x0) <= Node.minsize:
            self.type = Node.LEAF
        else:
            self.type = Node.BRANCH

    # Recursively subdivides a rectangle. Division occurs 
    # ONLY if the rectangle spans a "feature of interest".
    def subdivide(self):
        if self.type == Node.LEAF:
            return
        x0,y0,x1,y1 = self.rect
        h = (x1 - x0)/2
        rects = []
        rects.append( (x0, y0, x0 + h, y0 + h) )
        rects.append( (x0, y0 + h, x0 + h, y1) )
        rects.append( (x0 + h, y0 + h, x1, y1) )
        rects.append( (x0 + h, y0, x1, y0 + h) )
        
        for n in range(len(rects)):
            span = self.spans_feature(rects[n])
            
            if span == True:
                self.children[n] = self.getinstance(rects[n])
                self.children[n].subdivide() # << recursion
                
    # A utility proc that returns True if the coordinates of
    # a point are within the bounding box of the node.
    def contains(self, x, y):
        x0,y0,x1,y1 = self.rect
        if x >= x0 and x <= x1 and y >= y0 and y <= y1:
            return True
        return False
        
    # Sub-classes must override these two methods.
    def getinstance(self,rect):
        return Node(self,rect)            
    def spans_feature(self, rect):
        return False
  
class QuadTree():
    
    def __init__(self, rootnode, minrect):
        Node.minsize = minrect
        rootnode.subdivide() # constructs the network of nodes
        self.maxdepth = 1 # the "depth" of the tree
        self.leaves = []
        self.allnodes = []

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
        self.allnodes.append(node)
        if node.type == Node.LEAF:
            self.leaves.append(node)
            if node.depth > self.maxdepth:
                self.maxdepth = node.depth
        for child in node.children:
            if child != None:
                self.traverse(child) # << recursion

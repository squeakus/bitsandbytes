# Malcolm Kesson Dec 19 2012
from quadtree import Node, QuadTree
from distances import pnt2line
import random
import sys
import pygame

# Returns the length of a vector "connecting" p0 to p1.
# To avoid using the sqrt() function the return value is
# the length squared.
def dist_sqrd(p0, p1):
    x,y = p0
    X,Y = p1
    i,j = (X - x, Y - y)
    return i * i + j * j

def getedges(rect):
    x0,y0,x1,y1 = rect
    
    edges = ( ((x0,y0),(x1,y0)), # top
              ((x1,y0),(x1,y1)), # right
              ((x1,y1),(x0,y1)), # bottom
              ((x0,y1),(x0,y0))) # left
    return edges

def getpairs(rect):
    x0,y0,x1,y1 = rect
    x0 += 2
    y0 += 2
    x1 += 2
    y1 += 2
    
    x0 *= 150
    y0 *= 150
    x1 *= 150
    y1 *= 150

    edges = ( (x0,y0),(x1,y0), # top
              (x1,y0),(x1,y1), # right
              (x1,y1),(x0,y1), # bottom
              (x0,y1),(x0,y0)) # left
    return edges


class CNode(Node):
    # Overrides the base class method.
    # Ensures Node.subdivide() uses instances of our custom 
    # class rather than instances of the base 'Node' class.
    def getinstance(self,rect):
        return CNode(self,rect)
    
    # Overrides the base class method.
    # Test if the vertices of a rectangle spans the circumference 
    # of a circle(s). To avoid sampling errors the proc returns True
    # if any edge lies within the radius of any circle. However, the
    # 'edge test' is applied only to rectangles whose parent node has
    # a depth of recursion less than a specific (arbitary) value.
    def spans_feature(self, rect):
        x0,y0,x1,y1 = rect
        if self.depth < 3: # this may require adjustment
            for circle in CQuadTree.circles:
                rad,x,y = circle
                edges = getedges(rect)
                for edge in edges:
                    dist,loc = pnt2line( (x,y), edge[0], edge[1] )
                    if dist <= rad:
                        return True
        verts = [(x0,y0),(x0,y1),(x1,y1),(x1,y0)]
        
        for circle in CQuadTree.circles:
            rad,x,y = circle
            rad_sqrd = rad * rad
            center = (x,y)
            span = 0
            
            for vert in verts:
                d = dist_sqrd(vert,center)
                span += (d <= rad_sqrd)
            if span > 0 and span < 4:
                return True
        return False
         
class CQuadTree(QuadTree):
    circles = []   # list of tuples (rad,x,y)

    def __init__(self, rootnode, minrect, circles):
        CQuadTree.circles = circles
        QuadTree.__init__(self, rootnode, minrect)

if __name__=="__main__":
    pygame.init()
    window = pygame.display.set_mode((640, 640)) 

    rootrect = [-2.0, -2.0, 2.0, 2.0]
    resolution = 0.1
  
    circles = []
    random.seed(1)
    for n in range(15):
        r = random.uniform(0.2, 0.8)
        x = random.uniform(-2.0, 2.0)
        y = random.uniform(-2.0, 2.0)
        circles.append( (r,x,y) )

    for i in range(250):
        newcircles = []
        for circle in circles:
            r,x,y = circle
            y += 0.01
            newcircles.append((r,x,y))
        circles = newcircles
        
        rootnode = CNode(None, rootrect)
        tree = CQuadTree(rootnode, resolution, circles)
        print "finished quadtree", i

        window.fill((0,0,0))
        for idx,node in enumerate(tree.leaves):
            pygame.draw.polygon(window, (255,255,255), getpairs(node.rect), 1)
            pygame.display.flip()
        name = "img%04d.png" % i
        pygame.image.save(window, name)
        
    print('Wrote %d polygons' % len(tree.leaves))

    while True: 
        for event in pygame.event.get(): 
            if event.type == pygame.QUIT: 
                sys.exit(0) 

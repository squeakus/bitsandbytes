# cquadtree.py
# A relatively simple example of using the Node and QuadTree 
# class.
# Malcolm Kesson Dec 19 2012
from quadtree import Node, QuadTree
from distances import pnt2line
import random
import sys
import pygame


#____UTILITY PROCS_______________________________________
# Returns the length of a vector "connecting" p0 to p1.
# To avoid using the sqrt() function the return value is
# the length squared.
def dist_sqrd(p0, p1):
    x,y,z = p0
    X,Y,Z = p1
    i,j,k = (X - x, Y - y, Z - z)
    return i * i + j * j + k * k
#_______________________________________________________
def getedges(rect):
    x0,z0,x1,z1 = rect
    edges = ( ((x0,0,z0),(x1,0,z0)), # top
              ((x1,0,z0),(x1,0,z1)), # right
              ((x1,0,z1),(x0,0,z1)), # bottom
              ((x0,0,z1),(x0,0,z0))) # left
    return edges
#_______________________________________________________
  
class CNode(Node):
    #_______________________________________________________
    # Overrides the base class method.
    # Ensures Node.subdivide() uses instances of our custom 
    # class rather than instances of the base 'Node' class.
    def getinstance(self,rect):
        return CNode(self,rect)
    
    #_______________________________________________________
    # Overrides the base class method.
    # Test if the vertices of a rectangle spans the circumference 
    # of a circle(s). To avoid sampling errors the proc returns True
    # if any edge lies within the radius of any circle. However, the
    # 'edge test' is applied only to rectangles whose parent node has
    # a depth of recursion less than a specific (arbitary) value.
    def spans_feature(self, rect):
        x0,z0,x1,z1 = rect
        if self.depth < 3: # this may require adjustment
            for circle in CQuadTree.circles:
                rad,x,y,z = circle
                edges = getedges(rect)
                for edge in edges:
                    dist,loc = pnt2line( (x,0,z), edge[0], edge[1] )
                    if dist <= rad:
                        return True
        verts = [(x0,0,z0),(x0,0,z1),(x1,0,z1),(x1,0,z0)]
        for circle in CQuadTree.circles:
            rad,x,y,z = circle
            rad_sqrd = rad * rad
            center = (x,y,z)
            span = 0
            for vert in verts:
                d = dist_sqrd(vert,center)
                span += (d <= rad_sqrd)
            if span > 0 and span < 4:
                return True
        return False
         
class CQuadTree(QuadTree):
    circles = []   # list of tuples (rad,x,y,z)
    #_______________________________________________________
    def __init__(self, rootnode, minrect, circles):
        CQuadTree.circles = circles
        QuadTree.__init__(self, rootnode, minrect)
        
#_______________________________________________________
# Returns a string containing the rib statement for a
# four sided polygon positioned at height "y".
def RiPolygon(rect, y):    
    x0,z0,x1,z1 = rect
    verts = []
    verts.append(' %1.3f %1.3f %1.3f' % (x0,y,z0))
    verts.append(' %1.3f %1.3f %1.3f' % (x0,y,z1))
    verts.append(' %1.3f %1.3f %1.3f' % (x1,y,z1))
    verts.append(' %1.3f %1.3f %1.3f' % (x1,y,z0))
    rib =  '\tPolygon "P" ['
    rib += ''.join(verts)
    rib += ']\n'
    return rib
#_______________________________________________________
  
if __name__=="__main__":
    pygame.init()
    window = pygame.display.set_mode((640, 480)) 

    rootrect = [-2.0, -2.0, 2.0, 2.0]
    resolution = 0.02
  
    circles = []
    random.seed(1)
    for n in range(17):
        r = random.uniform(0.2, 0.8)
        x = random.uniform(-2.0, 2.0)
        z = random.uniform(-2.0, 2.0)
        circles.append( (r,x,0,z) )
        
    #circles = [(1.9,0,0,0),(1.0,0.95,0,0)]
    rootnode = CNode(None, rootrect)
    tree = CQuadTree(rootnode, resolution, circles)
    
    # Output RenderMan polygons for each node
    ribpath = 'leaves.rib'
    f = open(ribpath,'w')
    f.write('AttributeBegin\n')
    for node in CQuadTree.allnodes:
        height = node.depth * 0.1
        if node.depth == CQuadTree.maxdepth:
            f.write('\tColor 0 .5 0\n')
        else:
            f.write('\tColor 1 1 1\n')
        print node.rect,height
        f.write(RiPolygon(node.rect, height))
        pygame.draw.polygon(window, (0,0,255), node.rect, 1)
        pygame.display.flip() 


        
    f.write('AttributeEnd\n')
    f.close()
    print('Wrote %d polygons' % len(CQuadTree.leaves))



# #draw a line - see http://www.pygame.org/docs/ref/draw.html for more 
# pygame.draw.line(window, (255, 255, 255), (0, 0), (30, 50))

# #draw it to the screen
# pygame.display.flip() 

# #input handling (somewhat boilerplate code):
# while True: 
#    for event in pygame.event.get(): 
#       if event.type == pygame.QUIT: 
#           sys.exit(0) 
#       else: 
#           print event 

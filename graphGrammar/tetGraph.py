import networkx as nx
import matplotlib.pyplot as plt
import analyser
from geometry import *
from decimal import *
from math import *
import graph

class tetGraph():
    def __init__(self):
        self.g = graph.graph()

    def check_dist(self,nodeList):
        for nodeA in nodeList:
            for nodeB in nodeList:
                if not nodeA == nodeB: 
                    pt1 = self.g.node[nodeA]["xyz"]
                    pt2 = self.g.node[nodeB]["xyz"]
                    print nodeA,"->",nodeB,":",str(euclidean_distance(pt1,pt2))

    def euclid(self,p, q):
        return round(sqrt(sum([(p[i] - q[i]) **2 for i in range(len(p))])),3)

    def midpoint(self,ptA,ptB):
        return [(ptB[0]+ptA[0])/2,(ptB[1]+ptA[1])/2,(ptB[2]+ptA[2])/2]
    
    def connect_neighbours(self,nodeList,length):
        for nodeA in nodeList:
            for nodeB in nodeList:
                pt1 = self.node[nodeA]["xyz"]
                pt2 = self.node[nodeB]["xyz"]
                if self.distance(pt1,pt2) == length:
                    self.g.add_edge(nodeA,nodeB)
                    self.g.save_picture()

    # creates a regular tetrahedron(4 equilateral triangles)
    def square(self,start,length):
        length= length*1000
        x0,y0,z0=start[0],start[1],start[2]
        pt0=int(x0),int(y0),int(z0)

        x1,y1,z1= int(x0+length),int(y0),int(z0)
        pt1=x1,y1,z1

        x2,y2,z2= int(x0+length),int(y0+length),int(z0)
        pt2=x2,y2,z2

        x3,y3,z3= int(x0),int(y0+length),int(z0)
        pt3=x3,y3,z3

        mid = self.midpoint(pt0,pt2)
        height = int((sqrt(2)*length)/2)
        x4,y4,z4= int(mid[0]),int(mid[1]),int(z0+height)
        pt4=x4,y4,z4

        nodeList=[pt0,pt1,pt2,pt3,pt4]
        idList=[]
        for node in nodeList:
          idList.append(self.g.add_unique_node(node,"test"))  
        self.connect_neighbours(idList,length)
        self.check_dist(idList)
        return nodeList

    def regular(self,start,length):
        length = length*1000
        #use the first coordinate to define the rest
        x0,y0,z0=start[0],start[1],start[2]
        pt0=[int(x0),int(y0),int(z0)]

        x1,y1,z1= int(x0+length),int(y0),int(z0)
        pt1=[x1,y1,z1]

        x2,y2,z2=int(x0+length/2),int(y0+sqrt(length**2-((length/2)**2))),int(z0)
        pt2=[x2,y2,z2]

        height=int(start[2]+(sqrt(6)/3)*length)
        x3,y3,z3=int(x2),int(y0+tan(0.523598776)*(length/2)),height #don't forget to use rads
        pt3=[x3,y3,z3]
        
        nodeList=[pt0,pt1,pt2,pt3]
        idList=[]
        for node in nodeList:
            if node[2] == 0:
                idList.append(self.g.add_unique_node(node,"fixed")) 
            else:
                idList.append(self.g.add_unique_node(node,"nowt"))  

        self.g.connect_all_nodes(idList)
        return nodeList

    #4 tetrahedra
    def fourtet(self,start,length):
        tet0 = self.regular(start,length)
        tet1 = self.regular(tet0[1],length)
        tet2 = self.regular(tet0[2],length)
        tet3 = self.regular(tet0[3],length)
        
        vertices = [tet0[0],tet1[1],tet2[2],tet3[3]]
        return vertices

    def multi4(self,func,start,length):
        nodes1 = func(start,length)
        func(nodes1[1],length)
        func(nodes1[2],length)
        func(nodes1[3],length)
        func(nodes1[4],length)
    

    #attempt at base
    def four_aligned(self,start,length):
        nodeList = self.regular(start,length)
        self.regular(nodeList[2],length)
        nodeList = self.regular(nodeList[1],length)
        self.regular(nodeList[2],length)
        nodeList = self.regular(nodeList[1],length)
        self.regular(nodeList[2],length)
        nodeList = self.regular(nodeList[1],length)

    def eight_aligned(self,start,length):
        nodeList = self.regular(start,length)
        self.regular(nodeList[2],length)
        nodeList = self.regular(nodeList[1],length)
        self.regular(nodeList[2],length)
        nodeList = self.regular(nodeList[1],length)
        self.regular(nodeList[2],length)
        nodeList = self.regular(nodeList[1],length)


    def base(self,start,length):
        rowCount=1
        row1 = self.regular(start,length)
        row2 = self.regular(row1[2],length)
        row3 = self.regular(row2[2],length)
        row4 = self.regular(row3[2],length)

        rows = [row1,row2,row3,row4]
        
        for row in rows:
            for i in range(rowCount):
                row = self.regular(row[1],length)

    def multi_base(self,start,length):
        nodeList = self.regular(start,length)
        row1 = self.regular(nodeList[1],length)
        row2 = self.regular(nodeList[2],length)
        row3 = self.regular(nodeList[3],length)
        row1 = self.regular(row1[1],length)
        row1 = self.regular(row1[1],length)
        row2 = self.regular(row2[1],length)
        row2 = self.regular(row2[1],length)
        row3 = self.regular(row3[3],length)
        row3 = self.regular(row3[1],length)

    def octet(self,start,length):
        nodeList = self.regular(start,length)
        row1 = self.regular(nodeList[1],length)
        row2 = self.regular(nodeList[2],length)
        row3 = self.regular(nodeList[3],length)
        row1 = self.regular(row1[1],length)
        row1 = self.regular(row1[1],length)
        self.fourtet(row2[1],length)
        row2 = self.regular(row2[1],length)
        self.fourtet(row3[3],length)

    def levels(self,origin,length,levels):
        length = length * 1000
        x,y,z = origin
        for i in range(levels):
            tetra.base([x,y,z],3)
            print "X:",x
            x=int(x+(length/2))
            y=int(y+tan(0.523598776)*(length/2))
            z=int(z+(sqrt(6)/3)*length)
            

origin = (0,0,0)
length = 3
tetra = tetGraph()
tetra.regular(origin, length)
#tetra.levels(origin,length,3)


verts = tetra.fourtet(origin,length)
tetra.fourtet(verts[1],length)
tetra.fourtet(verts[2],length)
tetra.fourtet(verts[3],length)
tetra.octet(origin,length)
#tetra.square([0,0,3],3)
tetra.multi4(tetra.square,origin,3)
#tetra.four_aligned(origin,length)


print "final frame: ",str(tetra.g.nodeCounter-1)

#will it blend?
analyser = analyser.Analyser('test',"moo",True)
analyser.myGraph=tetra.g
analyser.parse_graph(tetra.g)
analyser.apply_stresses()
analyser.create_slf_file()
analyser.test_slf_file()
analyser.parse_results()
analyser.print_stresses()
analyser.show_analysis()

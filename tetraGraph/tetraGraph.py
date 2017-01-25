import networkx as nx
from math import *
import os,time
import analyser

class tetraGraph(nx.Graph):
    
    def __init__(self, *args, **kwargs):
        super(tetraGraph, self).__init__(*args, **kwargs)
        self.popFolder=os.getcwd()+"/population/"
        self.startTime=time.time()
	self.save=True
	self.frameCount=0

    # The order of a graph is the number of nodes.
    def get_unused_nodeid(self):
        return self.order()

    #returns euclidean distance
    def distance(self,p, q):
        dist=int(round(sqrt(sum([(p[i]-q[i])**2 for i in range(len(p))])),0))
	return dist

    def print_dist(self,nodeList):
        for nodeA in nodeList:
            for nodeB in nodeList:
                if not nodeA == nodeB: 
                    pt1 = self.node[nodeA]["xyz"]
                    pt2 = self.node[nodeB]["xyz"]
                    print nodeA,"->",nodeB,":",str(self.distance(pt1,pt2))

    def midpoint(self,ptA,ptB):
        x,y,z=(ptB[0]+ptA[0])/2,(ptB[1]+ptA[1])/2,(ptB[2]+ptA[2])/2
        mid=[int(round(x,0)),int(round(y,0)),int(round(z,0))]
        return mid

    def mirror(self,pts,axis):
        retval = list()
        for pt in pts:
            if axis == "x":
                inverse1,inverse2,inverse3 = -pt[0],pt[1], pt[2]
            if axis == "y":
                inverse1,inverse2,inverse3 = pt[0],-pt[1], pt[2]
            if axis == "z":
                inverse1,inverse2,inverse3 = pt[0],pt[1], -pt[2]
            inverse = inverse1,inverse2,inverse3
            retval.append(inverse)
        pts.reverse()
        for pt in pts:
            retval.append(pt)
        return retval

    def offset(self,pts,offset,axis):
        retval = list()
        for pt in pts:
            if axis == "x":
                inverse1,inverse2,inverse3 = pt[0]+offset,pt[1], pt[2]
            if axis == "y":
                inverse1,inverse2,inverse3 = pt[0],pt[1]+offset, pt[2]
            if axis == "z":
                inverse1,inverse2,inverse3 = pt[0],pt[1], pt[2]+offset
            inverse = inverse1,inverse2,inverse3
            retval.append(inverse)
        return retval

    # if it already exists, just return that node ID
    def add_unique_node(self,coords, nodeType):
        new = True
        ptA =[int(coords[0]),int(coords[1]),int(coords[2])]
        for node in self.node:
            ptB = self.node[node]['xyz']
            ptB = [int(ptB[0]),int(ptB[1]),int(ptB[2])]
            if ptA == ptB:
                new = False
                id = node
                break
        if new:
            id = self.get_unused_nodeid()
            self.add_node(id, xyz= ptA, label = nodeType)
        return id

    #connect all nodes within a given range
    def connect_neighbours(self,nodeList,length):
        for nodeA in nodeList:
            for nodeB in nodeList:
                if not self.has_edge(nodeA,nodeB):
                    pt1 = self.node[nodeA]["xyz"]
                    pt2 = self.node[nodeB]["xyz"]
                    if int(self.distance(pt1,pt2))==length:
                        self.add_edge(nodeA,nodeB)
                        self.save_graph()

    #generates a .mesh file from the graph
    def create_mesh(self,name):
        nodeList=[]
	edgeList=[]
        for nodeID in self.nodes():
            node={}
            xyz =self.node[nodeID]['xyz']
	    x,y,z=xyz[0],xyz[1],xyz[2]
            label=self.node[nodeID]['label']
            node={'id':str(node),'x':x,'y':y,'z':z,'label':label}
            nodeList.append(node)
        for idx,edge in enumerate(self.edges_iter()):
            edge = {'id':idx,'ptA':edge[0],'ptB':edge[1]}
            edgeList.append(edge)   
        
        filename=name+'.mesh' 
        mesh= open(filename,'w')
        mesh.write("MeshVersionFormatted 1\nDimension\n3 \n")
        mesh.write("Vertices\n"+str(len(nodeList))+" \n")
        for node in nodeList:
            mesh.write(str(node['x'])+" "+str(node['y'])+" "+str(node['z'])+" 0  \n")
        mesh.write("Edges\n"+str(len(edgeList))+" \n")
        for edge in edgeList:
            ptA, ptB = int(edge['ptA']),int(edge['ptB'])
            mesh.write(str(ptA+1)+" "+str(ptB+1)+" 0 \n")
        mesh.write("End\n")
        mesh.close()

    def save_graph(self,name=None):
        if self.save:
            if name == None:
                filename = self.popFolder+"img%04d"%(self.frameCount)
            else:
                filename = name
            self.create_mesh(filename)
            self.frameCount += 1

    #creates a regular tetrahedron(4 equilateral triangles)
    def regular(self,origin,length):
        #use the first coordinate to define the rest
        height=int(origin[2]+(sqrt(6)/3)*length)
        x0,y0,z0=int(origin[0]),int(origin[1]),int(origin[2])
        pt0=[x0,y0,z0]
        x1,y1,z1= int(x0+length),int(y0),int(z0)
        pt1=[x1,y1,z1]
        x2,y2,z2=int(x0+length/2),int(y0+sqrt(length**2-((length/2)**2))),z0
        pt2=[x2,y2,z2]
        x3,y3,z3=int(x2),int(y0+tan(0.523598776)*(length/2)),height #don't forget to use rads
        pt3=[x3,y3,z3]
        nodeList=[pt0,pt1,pt2,pt3]
        #add pts as nodes to a graph
        idList=[]
        for node in nodeList:
            if node[2] == 0:
                idList.append(self.add_unique_node(node,"fixed")) 
            else:
                idList.append(self.add_unique_node(node,"load"))  
        self.connect_neighbours(idList,length)
        return nodeList

    # creates a square tetrahedron(3 equilateral triangles,square base)
    def square(self,origin,length):
        #define points starting from origin
        height = int((sqrt(2)*length)/2)
        x0,y0,z0=int(origin[0]),int(origin[1]),int(origin[2])
        pt0=[x0,y0,z0]
        x1,y1,z1= int(x0+length),int(y0),int(z0)
        pt1=[x1,y1,z1]
        x2,y2,z2= int(x0+length),int(y0+length),int(z0)
        pt2=[x2,y2,z2]
        x3,y3,z3= int(x0),int(y0+length),int(z0)
        pt3=[x3,y3,z3]
        mid = self.midpoint(pt0,pt2)
        x4,y4,z4= int(mid[0]),int(mid[1]),int(z0+height)
        pt4=[x4,y4,z4]
        # add points to the graph
        nodeList=[pt0,pt1,pt2,pt3,pt4]
        idList=[]
        for node in nodeList:
            if node[2] == 0:
                idList.append(self.add_unique_node(node,"fixed")) 
            else:
                idList.append(self.add_unique_node(node,"load"))   
        self.connect_neighbours(idList,length)
        return nodeList

    def octahedron(self,origin,length):
        idList=[]
        nodes = self.square(origin,length)
        mirrored = self.mirror(nodes,'z')
        for node in mirrored:
            idList.append(self.add_unique_node(node,"load"))   
        self.connect_neighbours(idList,length)
        points = nodes+mirrored
        return points

    def fourtet(self,func,origin,length):
        print(func)
        nodes = func(origin,length)
        print(nodes)
        for i in range(1,len(nodes)):
            func(nodes[i],length)

    def grid(self,func,origin,length,size):
        current=origin
        floorNodes=[]
        for x in range(size):
            print "row",x
            for y in range(size):
                func(current,length)
                current = [current[0],current[1]+length,origin[2]]
            current = [current[0]+length,origin[1],current[2]]

        for nodeId in self.nodes():
            floorNodes.append(self.node[nodeId]['xyz'])
            
        zOffset = int(sqrt(2)*length)
        
        for z in range(size/2):
            print "floor",z
            floorNodes= self.offset(floorNodes,zOffset,'z')
            print "noOfFloorNodes:",len(floorNodes)
            idList=[]
            for node in floorNodes:
                idList.append(self.add_unique_node(node,"load"))  
            print "idList Length",len(idList)
            self.connect_neighbours(idList,length)

length = 1000 #make it big to keep it in integers
height = int((sqrt(2)*length)/2)
origin = [0,0,0]
tetra = tetraGraph()
tetra.grid(tetra.octahedron,origin,length,6)
#tetra.octahedron(origin,length)
#tetra.fourtet(tetra.regular,origin,length)
#tetra.fourtet(tetra.octahedron,origin,length)

print "nodes:",tetra.number_of_nodes(),"edges",tetra.number_of_edges()
print "total time",str(time.time()-tetra.startTime)
print "final frame: ",str(tetra.frameCount-1)

#will it blend?
azr = analyser.Analyser('test',"moo",True)
azr.myGraph=tetra
azr.parse_graph(tetra)
azr.apply_stresses()
azr.create_slf_file()
azr.test_slf_file()
azr.parse_results()
azr.print_stresses()
azr.show_analysis()

#! /usr/bin/env python

import re,subprocess
meshName="test.mesh"
totalLoad=1000000
avrLoad=0 
xMax=0
yMax=0
zMax=0 
xMin=0
yMin=0
zMin=0 

edgeList=[]
triList=[]
nodeList=[]
fixedList=[]
loadList=[]
loadNodes=[]
loadElems=[]
stressList=[]

# variables for the material being used
width = 100
height = 200
Emod = 10000000000
density = 5300
area = (float(width))*(float(height))*10.0**(-6)
iz = (((float(width))*(float(height)**3))/12)*10**(-12)
iy = (((float(height))*(float(width)**3))/12)*10**(-12)

#reads nodes and edges and adds them to the lists in the form
#edge(id,ptA,ptB) and node(id,x,y,z)
def readMesh():
    global xMax,yMax,zMax,xMin,yMin,zMin
    # getting no. of elements from file header
    meshFile = open(meshName,'r')
    lines = iter(meshFile)
    for line in lines:
        #extract nodes
        if line.strip() == 'Vertices':
            line = lines.next()
            nodes = int(line)
            for i in range(nodes):
                line = lines.next()
                nodeString = line.strip()
#                regex = "[+-]?\d+\.\d+[e][\-][\d][\d]|[+-]?\d+\.\d+|\d+|[+-]?\d+"
                regex = "[+-]?\d+[\.\d|\.e\-\d]*"
                #regex = "\d+\.\d+|\d+"
                match = re.findall(regex, nodeString)
                x,y,z = float(match[0]),float(match[1]),float(match[2])
                if x > xMax:
                    xMax = x
                elif x < xMin:
                    xMin = x
                if y > yMax:
                    yMax = y
                elif y <yMin:
                    yMin = y
                if z >zMax:
                    zMax = z
                elif z < zMin:
                    zMin = z
                
                node = {'id':str(i),'x':str(x),'y':str(y),'z':str(z)}
                nodeList.append(node)            

        #extract triangles
        if line.strip() == 'Triangles':
            line = lines.next()
            triangles = int(line)
            for i in range(triangles):
                line = lines.next()
                triString = line.strip()
                regex = "\d+"
                match = re.findall(regex, triString) 
                #slffea freaks out it the index does not start at 0
                a,b,c = int(match[0])-1,int(match[1])-1,int(match[2])-1 
                triangle = {'id':str(i),'ptA':str(a),'ptB':str(b),'ptC':str(c)}
                triList.append(triangle)
            print "triangles", len(triList)

        #extract edges
        if line.strip() == 'Edges':
            line = lines.next()
            edges = int(line)
            for i in range(edges):
                line = lines.next()
                edgeString = line.strip()
                regex = "\d+"
                match = re.findall(regex, edgeString) 
                a,b = int(match[0])-1,int(match[1])-1
                edge = {'id':str(i),'ptA':str(a),'ptB':str(b)}
                edgeList.append(edge)
            print "edges",len(edgeList)
    meshFile.close()
    build_lists()


def build_lists():
    global nodeList,xMax,yMax,zMax,avrLoad
    #turn triangles into edges
    for tri in triList:
        edge1 = [tri['ptA'],tri['ptB']]
        edge2 = [tri['ptB'],tri['ptC']]
        edge3 = [tri['ptC'],tri['ptA']]
        newEdge1 = True
        newEdge2 = True
        newEdge3 = True
        for edge in edgeList:
            if newEdge1 and edge['ptA'] in edge1 and edge['ptB'] in edge1:
                newEdge1 = False
            if newEdge2 and edge['ptA'] in edge2 and edge['ptB'] in edge2:
                newEdge2 = False
            if newEdge3 and edge['ptA'] in edge3 and edge['ptB'] in edge3:
                newEdge3 = False
        if newEdge1:
            newEdge = {'id':str(len(edgeList)),'ptA':edge1[0],'ptB':edge1[1]}
            edgeList.append(newEdge)
        if newEdge2:
            newEdge = {'id':str(len(edgeList)),'ptA':edge2[0],'ptB':edge2[1]}
            edgeList.append(newEdge)
        if newEdge3:
            newEdge = {'id':str(len(edgeList)),'ptA':edge3[0],'ptB':edge3[1]}
            edgeList.append(newEdge)

    # removes the unwanted nodes and stuff by creating a new beamList upon which the nodeList will be based
    for edge in edgeList: # look at each edge
        for node in nodeList: # look at each node in each edge
            if edge['ptA'] == node['id']:
                edge['ptA'] = node['x'], node['y'], node['z'] # if the node exists, replace the node with its corresponding co-ordinates
            if edge['ptB'] == node['id']:
                edge['ptB'] = node['x'], node['y'], node['z']

    # Delete the old nodeList which contains unwanted nodes; build a new one.
    nodeList = [] 
    for edge in edgeList:
        if edge['ptA'] not in nodeList:            
            nodeList.append(edge['ptA'])
        if edge['ptB'] not in nodeList:
            nodeList.append(edge['ptB'])

    # Now we need to fix the edgeList and set it back to its original format
    for index,edge in enumerate(edgeList):
        edge['id'] = str(index)
        for i,n in enumerate(nodeList):
            if edge['ptA'] == n:
                edge['ptA'] = str(i)
            if edge['ptB'] == n:
                edge['ptB'] = str(i)                    

    # Re-structure the new nodeList so it can be read by everything else.
    newnodeList = []
    for i, node in enumerate(nodeList):
        x,y,z = node[0], node[1], node[2]
        adjustednode = {'id':str(i),'x':str(x),'y':str(y),'z':str(z)}
        newnodeList.append(adjustednode)
    nodeList = newnodeList 

    #build fixed points list and load list
    for node in nodeList:
        z = node['z']
        if float(z) == zMin:
            fixedList.append(node)
        elif float(z) == zMax:
            loadNodes.append(node['id'])
    print "bases: ",len(fixedList)
    print "loads: ",len(loadNodes)

    #build list of beams connecting Nodes bearing loads
    for beam in edgeList:
        if beam['ptA'] in loadNodes and beam['ptB'] in loadNodes:
            loadElems.append(beam)
    avrLoad = totalLoad/len(loadElems)
    print "loadEdges: ",len(loadElems)," with load: ",str(avrLoad)

def create_slf_file():
    mesh= open('mesh','w')
    mesh.write('numel numnp nmat nmode (This is for a beam bridge)\n')
    mesh.write(str(len(edgeList))+' '+str(len(nodeList))+' 1 0\n')
    mesh.write('matl no., E mod, Poiss. Ratio, density, Area, Iy, Iz\n')
    mesh.write('0 '+str(Emod)+' 0.0000 '+str(density)+' '+str(area)+' '+str(iy)+' '+str(iz) + '\n')
    mesh.write('el no.,connectivity, matl no, element type\n')
    for edge in edgeList:
        mesh.write(edge['id']+' '+str(edge['ptA'])+' '+str(edge['ptB'])+' 0 2 \n')
    mesh.write('node no., coordinates\n')
    for node in nodeList:
        mesh.write(node['id']+' '+node['x']+' '+node['y']+' '+node['z']+"\n")
    mesh.write("element with specified local z axis: x, y, z component\n -10\n")
    mesh.write('prescribed displacement x: node  disp value\n')
    for node in fixedList:
        mesh.write(node['id']+" 0.0\n")
    mesh.write('-10\nprescribed displacement y: node  disp value\n')
    for node in fixedList:
        mesh.write(node['id']+" 0.0\n")
    mesh.write('-10\nprescribed displacement z: node  disp value\n')
    for node in fixedList:
        mesh.write(node['id']+" 0.0\n")
    mesh.write('-10\nprescribed angle phi x: node angle value\n -10')
    mesh.write('\nprescribed angle phi y: node angle value\n -10')
    mesh.write('\nprescribed angle phi z: node angle value\n -10')
    mesh.write('\nnode with point load x, y, z and 3 moments phi x, phi y, phi z\n-10\n')
    mesh.write('element with distributed load in local beam y and z coordinates\n')
    for elem in loadElems:
        mesh.write(elem['id']+' 0 -'+str(avrLoad)+' \n') 
    mesh.write('-10\nelement no. and gauss pt. no. with local stress vector xx and moment xx,yy,zz\n-10')
    mesh.close()

def testMesh():
    #run the analysis
    print "running analysis"
    process = subprocess.Popen('echo mesh | bm', shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    process.communicate()
    #show analysises
    print "analysis complete, generating output"
    subprocess.Popen('echo mesh | bmpost', shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE)

def readStress():  
    maxStress = 0.0
    results = open('triangle.otr','r')  # opens the results file
    line = results.readline()
    while(line):
        if line.startswith("element no. and nodal pt. no. with local stress xx,xy,zx and moment xx,yy,zz") or line.startswith("element no. and gauss pt. no. with local stress xx,xy,zx and moment xx,yy,zz"): 
            print "found the stresses!"
            line = results.readline()
            while  line != ('-10'):
                line = results.readline()
                if line.startswith(' '):		
                    result = line.split()
                    stress = result[2:5]
                    for strValue in stress:
                        value = abs(float(strValue))
                        print value
                        if abs(float(value)) > maxStress:
                            print "found new max stress!",value
                            maxStress = abs(float(value))
                    print stress
                    stressList.append(stress)# Adds the stresses to the
                else:
                    break
        line =results.readline()
    print "The Max stress is ",maxStress
        

        
                
print "generating slffea file"
readMesh()
#writeMesh()
create_slf_file()
testMesh()
#readStress()




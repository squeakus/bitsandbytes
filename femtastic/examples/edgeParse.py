#! /usr/bin/env python

import re,subprocess
meshName="test.msh"
edgeList =[]
triList = []
nodeList =[]
fixedList =[]
loadList =[]
stressList =[]

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
    # getting no. of elements from file header
    meshFile = open(meshName,'r')
    header = meshFile.readline()
    header = header.split()
    nodes,triangles,edges = int(header[0]),int(header[1]),int(header[2])
    print "nodes: ",nodes," triangles: ",triangles,"edges: ",edges
    #extract nodes
    for i in range(nodes):        
        nodeString = meshFile.readline().strip()
        regex = "\d+\.\d+|\d+"
        match = re.findall(regex, nodeString) 
        x,y,z = match[0],match[1],str(0)
        node = {'id':str(i),'x':x,'y':y,'z':z}
        nodeList.append(node)
    for node in nodeList:
        print node
  
    #extract triangles
    for i in range(triangles):
        triString = meshFile.readline().strip()
        regex = "\d+"
        match = re.findall(regex, triString) 
        #slffea freaks out it the index does not start at 0
        a,b,c = int(match[0])-1,int(match[1])-1,int(match[2])-1 
        triangle = {'id':str(i),'ptA':str(a),'ptB':str(b),'ptC':str(c)}
        triList.append(triangle)

    #extract edges
    for i in range(edges):
        edgeString = meshFile.readline().strip()
        regex = "\d+"
        match = re.findall(regex, edgeString) 
        a,b = int(match[0])-1,int(match[1])-1
        edge = {'id':str(i),'ptA':str(a),'ptB':str(b)}
        edgeList.append(edge)

    #build fixed points list and load list
    for node in nodeList:
        if float(node['y']) == 0:
            fixedList.append(node)
        if float(node['y']) == 1:
            print "found load",node
            loadList.append(node)

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
            newEdge = edge = {'id':str(len(edgeList)),'ptA':edge1[0],'ptB':edge1[1]}
            edgeList.append(newEdge)
        if newEdge2:
            newEdge = edge = {'id':str(len(edgeList)),'ptA':edge2[0],'ptB':edge2[1]}
            edgeList.append(newEdge)
        if newEdge3:
            newEdge = edge = {'id':str(len(edgeList)),'ptA':edge3[0],'ptB':edge3[1]}
            edgeList.append(newEdge)
    meshFile.close()

def writeMesh():
    mesh= open("mesh",'w')
    mesh.write('numel numnp nmat nmode (This is for a beam bridge)\n')
    mesh.write(str(len(edgeList))+' '+str(len(nodeList))+' 1 0\n')
    mesh.write('matl no., E mod, Poiss. Ratio, density, Area, Iy, Iz\n')
    mesh.write('0 '+str(Emod)+' 0.0000 '+str(density)+' '+str(area)+' '+str(iy)+' '+str(iz) + '\n')
    mesh.write('el no.,connectivity, matl no, element type\n')
    for edge in edgeList:
        mesh.write(edge['id']+' '+edge['ptA']+' '+edge['ptB']+' 0 2 \n')
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
    for node in loadList:
        mesh.write(node['id']+'-1000 0\n') 
    mesh.write('-10\nelement no. and gauss pt. no. with local stress vector xx and moment xx,yy,zz\n-10')
    mesh.close()
    
def testMesh():
    #run the analysis
    subprocess.Popen('echo mesh | bm', shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    #show analysis
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
writeMesh()
testMesh()
#readStress()




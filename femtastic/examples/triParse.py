#! /usr/bin/env python

import re
import subprocess
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

#reads nodes and edges and adds them to the lists in the form
#edge(id,ptA,ptB) and node(id,x,y,z)
def readMesh():
    # getting no. of elements from file header
    meshFile = open(meshName,'r')
    header = meshFile.readline()
    header = header.split()
    nodes,triangles,edges = int(header[0]),int(header[1]),int(header[2])
    print "nodes: ",nodes," triangles: ",triangles,"edges: ",edges
    #add nodes
    for i in range(nodes):        
        nodeString = meshFile.readline().strip()
        regex = "\d+\.\d+|\d+"
        match = re.findall(regex, nodeString) 
        x,y,z = match[0],match[1],match[2] 
        node = {'id':i,'x':x,'y':y,'z':z}
        nodeList.append(node)
  
    #extract triangles
    for i in range(triangles):
        triString = meshFile.readline().strip()
        regex = "\d+"
        match = re.findall(regex, triString) 
        #slffea freaks out it the index does not start at 0
        a,b,c = int(match[0])-1,int(match[1])-1,int(match[2])-1 
        triangle = {'id':i,'ptA':str(a),'ptB':str(b),'ptC':str(c)}
        triList.append(triangle)
    #extract edges
    for i in range(edges):
        edgeString = meshFile.readline().strip()
        regex = "\d+"
        match = re.findall(regex, edgeString) 
        a,b = match[0],match[1]
        edge = {'id':triangles+i+1,'ptA':a,'ptB':b}
        edgeList.append(edge)
    for node in nodeList:
        if float(node['y']) == 0:
            fixedList.append(node)
        if float(node['x']) == 0:
            print "load", node
            loadList.append(node)
    meshFile.close()

def writeMesh():
    mesh= open("triangle",'w')
    mesh.write('numel numnp nmat nmode (This is for a beam bridge)\n')
    mesh.write(str(len(triList))+' '+str(len(nodeList))+' 1 0 0\n')
    mesh.write('matl no., E mod, Poiss. Ratio, density, Area, Iy, Iz\n')
    mesh.write('0 '+str(Emod)+' 0.0000 '+str(density)+ '\n')
    mesh.write('el no.,connectivity, matl no, element type\n')
    for tri in triList:
        mesh.write(str(tri['id'])+' '+tri['ptA']+' '+tri['ptB']+' '+tri['ptC']+' 0\n')
    mesh.write('node no., coordinates\n')
    for node in nodeList:
        mesh.write(str(node['id'])+' '+node['x']+' '+node['y']+"\n")
    #add in fixed points
    mesh.write('prescribed displacement x: node  disp value\n')  
    for node in fixedList:
        mesh.write(str(node['id'])+'  0.0\n')
    mesh.write('-10\nprescribed displacement y: node  disp value\n')
    for node in fixedList:
        mesh.write(str(node['id'])+'  0.0\n')
    #add in loaded points
    mesh.write('-10\nelement with point load and load vector in x,y\n')
    for node in loadList:
        mesh.write(str(node['id'])+' 0.0  -1000.0\n') 
    #mesh.write("3 0.0 -1000\n")
    mesh.write("-10\nnode no. with stress and stress vector in xx,yy,xy\n-10\n")
    mesh.close()
    
def testMesh():
    #run the analysis
    subprocess.Popen('echo triangle | tri', shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    #show analysis
    subprocess.Popen('echo triangle | trpost', shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE)

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
readStress()




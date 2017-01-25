#! /usr/bin/env python

import re
import subprocess
edgeName="/home/jonathan/Jonathan/programs/tet/square.1.edge"
nodeName = "/home/jonathan/Jonathan/programs/tet/square.1.node"
edgeList =[]
nodeList =[]
floorList =[]
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
    # add edges
    edgeFile = open(edgeName,'r')
    nodeFile = open(nodeName,'r')
    edges = int(edgeFile.readline())
    for i in range(edges):
        edgeString = edgeFile.readline().strip()
        regex = "\d+"
        match = re.findall(regex, edgeString)
        edgeId,a,b = int(match[0])-1,int(match[1])-1,int(match[2])-1 
        edge = {'id':str(edgeId),'ptA':str(a),'ptB':str(b)}
        edgeList.append(edge)

    #add nodes
    nodeHeader = nodeFile.readline().strip().split(" ")
    nodes = int(nodeHeader[0])
    for i in range(nodes):
        nodeString = nodeFile.readline().strip()
        regex = "\d+\.\d+|\d+"
        match = re.findall(regex, nodeString) 
        nodeId,x,y,z = int(match[0])-1,match[1],match[2],match[3] 
        node = {'id':str(nodeId),'x':x,'y':y,'z':z}
        nodeList.append(node)
        
    #specify floor
    for node in nodeList:
        if float(node['x']) == 0:
            floorList.append(node) 
    #specify load
    for node in nodeList:
        if float(node['x']) == 1:
            loadList.append(node)
 
    edgeFile.close()
    nodeFile.close()

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
    for node in floorList:
        mesh.write(node['id']+" 0.0\n")
    mesh.write('-10\nprescribed displacement y: node  disp value\n')
    for node in floorList:
        mesh.write(node['id']+" 0.0\n")
    mesh.write('-10\nprescribed displacement z: node  disp value\n')
    for node in floorList:
        mesh.write(node['id']+" 0.0\n")
    mesh.write('-10\nprescribed angle phi x: node angle value\n -10')
    mesh.write('\nprescribed angle phi y: node angle value\n -10')
    mesh.write('\nprescribed angle phi z: node angle value\n -10')
    mesh.write('\nnode with point load x, y, z and 3 moments phi x, phi y, phi z\n-10\n')
    mesh.write('element with distributed load in local beam y and z coordinates\n')
    for node in loadList:
        mesh.write(node['id']+' 0    -5000\n') 
    mesh.write('-10\nelement no. and gauss pt. no. with local stress vector xx and moment xx,yy,zz\n-10')
    mesh.close()

def testMesh():
    #run the analysis
    subprocess.Popen('echo mesh | bm', shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE).wait()
    #show analysis
    subprocess.Popen('echo mesh | bmpost', shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE)

def readStress():  
    maxStress = 0.0
    results = open('mesh.obm','r')  # opens the results file
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
        
def newReadStress():
    maxStress = 0.0
    results = open('mesh.obm','r')  # opens the results file
    lines = iter(results)
    for line in lines:
        if line.startswith("element no. and nodal pt. no. with local stress xx,xy,zx and moment xx,yy,zz") or line.startswith("element no. and gauss pt. no. with local stress xx,xy,zx and moment xx,yy,zz"): 
            print "found the stresses!"
            lines.next()
            while  line.strip() != ('-10'):
                line = lines.next()
		print line
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
    print "The Max stress is ",maxStress
    results.close()

        
                
print "generating slffea file"
readMesh()
writeMesh()
testMesh()
#readStress()
newReadStress()




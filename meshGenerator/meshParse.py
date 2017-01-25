#! /usr/bin/env python

import re,subprocess
meshName="working.mesh"
totalLoad=10000
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
    global xMax,yMax,zMax,xMin,yMin,zMin,edgeList
    # getting no. of elements from file header
    meshFile = open(meshName,'r')
    for line in meshFile:
        result = line.split()
        if result[0]=='node':
            node = {'id':str(result[1]),'x':str(result[2]),'y':str(result[3]),'z':str(result[4]),'nType':result[5]}
            nodeList.append(node)
        if result[0]=='edge':
            edge = {'id':str(result[1]),'ptA':str(result[2]),'ptB':str(result[3])}
            edgeList.append(edge)
    
    nSize = len(nodeList)
    cnt =0
    edgeID = 0

    eSize = len(edgeList)
    print "nodeSize:",nSize
    print "edgeSize:",eSize
    for nodeA in nodeList:
        for nodeB in nodeList:
            if nodeA['x']==nodeB['x'] and nodeA['y']==nodeB['y'] and nodeA['z']==nodeB['z']:
                if nodeA['id'] != nodeB['id']:
                    print "DUPLICATED NODE",nodeA['id'],nodeB['id']
                    print 'x',nodeA['x'],"y",nodeA['y'],"z",nodeA['z']
                    print 'x',nodeA['x'],"y",nodeA['y'],"z",nodeA['z']
    for i in range(nSize):
        foundIdx = False
        if nodeList[i]['id'] != str(i):
            print "FUCKUP NODE AT ",i
    for i in range(eSize):
        foundIdx = False
        if edgeList[i]['id'] != str(i):
            print "FUCKUP EDGE AT ",i
    
    for i,edge in enumerate(edgeList):
        if edge['ptA'] == edge['ptB']:
            print "IDENTICAL POINTS IN EDGE",i
    #edgeList = edgeList[0:99]
    print "newLength:",len(edgeList)
    for node in nodeList:
            connected = False;
            for edge in edgeList:
                if node['id'] == edge['ptA'] or node['id'] == edge['ptB']:
                    connected = True
            if connected == False:
                print "FUCKING FLOATING NODE"

    apply_stresses()

def apply_stresses():
    global avrLoad
    #build fixed points list and load list
    for node in nodeList:
        x = float(node['x'])
        if  x == 0 or x==30.0:
            fixedList.append(node)
        #if node['nType'] == 'walkway':
        #    self.fixedList.append(node)
    #    if node['nType'] == 'walkway':
    #        loadNodes.append(int(node['id']))
    print "fixed:",len(fixedList)
    print "loaded",len(loadNodes)

    #SLFFEA applies load to edges, find edges connecting loadNodes
    for edge in edgeList:
        ptA,ptB = int(edge['ptA']),int(edge['ptB'])
        if ptA in loadNodes and ptB in loadNodes:
            loadElems.append(edge)

    #build list of beams connecting Nodes bearing loads
    for beam in edgeList:
        if beam['ptA'] in loadNodes and beam['ptB'] in loadNodes:
            loadElems.append(beam)
    #avrLoad = totalLoad/len(loadElems)
    avrLoad =1
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




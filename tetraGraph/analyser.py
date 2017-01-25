import subprocess,re,os
#import tetraGraph
from math import *

# global class variables
showAnalysis = False
defaultFitness = 1000000

# An unpleasant limitation in Python is the distinction between
# eval and exec. The former can only be used to return the value
# of a simple expression (not a statement) and the latter does not
# return anything.
def eval_or_exec(s):
    s = python_filter(s)
    try:
        retval = eval(s)
    except SyntaxError:
        # SyntaxError will be thrown by eval() if s is compound,
        # ie not a simple expression, eg if it contains function
        # definitions, multiple lines, etc. Then we must use
        # exec(). Then we assume that s will define a variable
        # called "XXXeval_or_exec_outputXXX", and we'll use that.
        exec(s)
        retval = XXXeval_or_exec_outputXXX
    return retval

# Create correct python syntax. We use {} to track indentation,
# which is not ideal because of the clash with dict literals.
def python_filter(txt):
    counter = 0
    for char in txt:
        if char == "{":
            counter += 1
        elif char == "}":
            counter -= 1
        tabstr = "\n" + "  " * counter
        if char == "{" or char == "}":
            txt = txt.replace(char, tabstr, 1)
    txt = "\n".join([line for line in txt.split("\n") 
                     if line.strip() != ""])
    return txt

def log_error(self,phenotype,msg):
            file = open('err.log.', 'a')
            file.write(msg)
            file.write(phenotype)
            file.close()


#this class will create the meshes and analyise them
class Analyser():
    broken=0
    def __init__(self,UID,program,save=False):
        self.name = 'testMesh'
        self.UID = UID
        #stores all the slffea values
        self.myGraph = None
        self.saveMesh = save
        self.program=program
        self.totalZLoad=10000
        self.totalYLoad=0
        self.avrZLoad=0
        self.avrYLoad=0
        self.edgeList=[]
        self.nodeList=[]
        self.fixedList=[]
        self.loadNodes=[]
        self.loadElems=[]
        self.max = {'x':0,'y':0,'z':0}
        self.min = {'x':0,'y':0,'z':0}
        self.XXStress = {'totalTens':1,'totalComp':1,'avrTens':0,'avrComp':0,'maxTens':0,'maxComp':0,'beamTens':1,'beamComp':1,'failTens':0,'failComp':0}
        self.XYStress = {'totalTens':1,'totalComp':1,'avrTens':0,'avrComp':0,'maxTens':0,'maxComp':0,'beamTens':1,'beamComp':1,'failTens':0,'failComp':0}
        self.ZXStress = {'totalTens':1,'totalComp':1,'avrTens':0,'avrComp':0,'maxTens':0,'maxComp':0,'beamTens':1,'beamComp':1,'failTens':0,'failComp':0}
        self.material ={}
        self.material['width'] = 100
        self.material['height'] = 200
        self.material['emod'] = 10000000000
        self.material['density'] = 5300
        self.material['area'] = (float(self.material['width']))*(float(self.material['height']))*10.0**(-6)
        self.material['iz'] = (((float(self.material['width']))*(float(self.material['height'])**3))/12)*10**(-12)
        self.material['iy'] = (((float(self.material['height']))*(float(self.material['width'])**3))/12)*10**(-12)
        self.material['allowedXXTension'] = (1.8*10**7)
        self.material['allowedXXCompression'] = (2.3*10**7)
        self.material['allowedXYTension'] = (0.6*10**6)
        self.material['allowedXYCompression'] = (8.4*10**6)
        self.material['allowedZXTension'] = self.material['allowedXYTension']
        self.material['allowedZXCompression'] = self.material['allowedXYCompression']
        self.material['maxmoment'] = (3*10**7)
        
    def create_graph(self):
        self.myGraph = eval_or_exec(self.program)
        self.parse_graph(self.myGraph)
        
    def parse_graph(self,myGraph):
        for node in myGraph.nodes():
            #xyz =myGraph.get_node_data(node)
            xyz=myGraph.node[node]['xyz']
            label = myGraph.node[node]['label']
            node = {'id':str(node),'x':xyz[0],'y':xyz[1],'z':xyz[2],'label':label}
            self.nodeList.append(node)
        for idx,edge in enumerate(myGraph.edges_iter()):
            edge = {'id':str(idx),'ptA':str(edge[0]),'ptB':str(edge[1])}
            self.edgeList.append(edge)   
        if self.saveMesh == True:
            self.create_mesh()
        
    def create_mesh(self,name='indiv'):
        if name == 'indiv':
            filename = "./indiv."+str(self.UID)+".mesh"
        else:
            filename=name+'.mesh' 
        mesh= open(filename,'w')
        mesh.write("MeshVersionFormatted 1\nDimension\n3 \n")
        mesh.write("Vertices\n"+str(len(self.nodeList))+" \n")
        for node in self.nodeList:
            mesh.write(str(node['x'])+" "+str(node['y'])+" "+str(node['z'])+" 0  \n")
        mesh.write("Edges\n"+str(len(self.edgeList))+" \n")
        for edge in self.edgeList:
            ptA, ptB = int(edge['ptA']),int(edge['ptB'])
            #ptA, ptB = edge['ptA'],edge['ptB']
            mesh.write(str(ptA+1)+" "+str(ptB+1)+" 0 \n")
            #mesh.write(str(ptA)+" "+str(ptB)+" 0 \n")
        mesh.write("End\n")
        mesh.close()

    def create_debug(self):
        mesh = open("debug.mesh",'w')
        for node in self.nodeList:
            mesh.write('node '+node['id']+' '+str(node['x'])+' '+str(node['y'])+' '+str(node['z'])+' '+node['label']+'\n')
        for edge in self.edgeList:
            mesh.write('edge '+edge['id']+' '+edge['ptA']+' '+edge['ptB']+'\n')

    def apply_stresses(self):        
        #build fixed points list and load list
        for node in self.nodeList:
            x,y,z = node['x'],node['y'],node['z']
            if  x == 0 and z == 0:
                self.fixedList.append([node,True])
            if  x == 30 and z== 0:
                self.fixedList.append([node,False])
            if node['label'] == 'fixed':
                self.fixedList.append([node,False])
            if node['label'] == 'load':
                self.loadNodes.append(int(node['id']))

        #SLFFEA applies load to edges, find edges connecting loadNodes 
        for edge in self.edgeList:
            ptA,ptB = int(edge['ptA']),int(edge['ptB'])
            if ptA in self.loadNodes or ptB in self.loadNodes:
                  self.loadElems.append(edge['id'])
        print "loadNodes:",len(self.loadElems)
        self.avrZLoad = self.totalZLoad/len(self.loadElems)
        print "load",self.avrZLoad
        self.avrYLoad=0
        
    def create_slf_file(self):
        mesh= open(self.name,'w')
        mesh.write('numel numnp nmat nmode (This is for a beam bridge)\n')
        mesh.write(str(len(self.edgeList))+' '+str(len(self.nodeList))+' 1 0\n')
        mesh.write('matl no., E mod, Poiss. Ratio,density, Area, Iy, Iz\n')
        mesh.write('0 '+str(self.material['emod'])+' 0.0000 '+str(self.material['density'])+' '+str(self.material['area'])+' '+str(self.material['iy'])+' '+str(self.material['iz']) + '\n')
        mesh.write('el no.,connectivity, matl no, element type\n')
        for i, edge in enumerate(self.edgeList): 
            mesh.write(str(i)+' '+str(edge['ptA'])+' '+str(edge['ptB'])+' 0 2 \n')
        mesh.write('node no., coordinates\n')
        for node in self.nodeList:
            mesh.write(node['id']+' '+str(node['x'])+' '+str(node['y'])+' '+str(node['z'])+"\n")
        mesh.write("element with specified local z axis: x, y, z component\n -10\n")
        mesh.write('prescribed displacement x: node  disp value\n')
        for node in self.fixedList:
            if node[1] == True:
                mesh.write(node[0]['id']+" 0.0\n")
        mesh.write('-10\nprescribed displacement y: node  disp value\n')
        for node in self.fixedList:
            mesh.write(node[0]['id']+" 0.0\n")
        mesh.write('-10\nprescribed displacement z: node  disp value\n')
        for node in self.fixedList:
            mesh.write(node[0]['id']+" 0.0\n")
        mesh.write('-10\nprescribed angle phi x: node angle value\n -10')
        mesh.write('\nprescribed angle phi y: node angle value\n -10')
        mesh.write('\nprescribed angle phi z: node angle value\n -10')
        mesh.write('\nnode with point load x, y, z and 3 moments phi x, phi y, phi z\n-10\n')
        mesh.write('element with distributed load in local beam y and z coordinates\n')
        for elem in self.loadElems: 
            mesh.write(elem+' -'+str(self.avrYLoad)+' -'+str(self.avrZLoad)+' \n') 
        mesh.write('-10\nelement no. and gauss pt. no. with local stress vector xx and moment xx,yy,zz\n-10')
        mesh.close()

    def test_slf_file(self):
        #run the structural analysis software
        cmd = 'bm'
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
        process.communicate(self.name)

    def show_analysis(self):
        cmd = "echo "+self.name+'.obm | bmpost'
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
        process.communicate(self.name)

    def parse_results(self):
        stressHeader1 = "element no. and nodal pt. no. with local stress xx,xy,zx and moment xx,yy,zz"
        stressHeader2 = "element no. and gauss pt. no. with local stress xx,xy,zx and moment xx,yy,zz"
        stressList = []
        results = open(self.name+'.obm','r')  # opens the results file
        lines = iter(results)
        for line in lines:
            if line.startswith(stressHeader1) or line.startswith(stressHeader2): 
                line = lines.next()
                while  line.strip() != ('-10'):		
                    result = line.split()
                    # check if its busted
                    if 'nan' in result or '-nan' in result:
                        self.create_debug()
                        self.create_mesh("broke")
                        self.log_error("structural failure",self.program)
                        print "**************STRUCTURE FAILED****************"
                        xxFloat,xyFloat,zxFloat= defaultFitness, defaultFitness,defaultFitness
                        xx,xy,zx = int(xxFloat),int(xyFloat),int(zxFloat)
                        stress = {'xx':xx,'xy':xy,'zx':zx}
                        stressList.append(stress)
                        exit()
                        break
                    else:  
                        xxFloat,xyFloat,zxFloat = float(result[2]),float(result[3]),float(result[4])
                        xx,xy,zx = int(xxFloat),int(xyFloat),int(zxFloat)
                        stress = {'xx':xx,'xy':xy,'zx':zx}
                        stressList.append(stress)
                        line = lines.next()
        results.close()
        self.calculate_stresses(stressList)
    
    def calculate_stresses(self,stressList):
        for stress in stressList:
            xx,xy,zx = stress['xx'],stress['xy'],stress['zx']
            #calculate stresses on XX plane
            df = defaultFitness
            if xx == df or xy == df or zx == df:
                self.XXStress = {'totalTens':df,'totalComp':df,'maxTens':df,'maxComp':df,'beamTens':1,'beamComp':1,'failTens':df,'failComp':df}
                self.XYStress = {'totalTens':df,'totalComp':df,'maxTens':df,'maxComp':df,'beamTens':1,'beamComp':1,'failTens':df,'failComp':df}
                self.ZXStress = {'totalTens':df,'totalComp':df,'maxTens':df,'maxComp':df,'beamTens':1,'beamComp':1,'failTens':df,'failComp':df}
            if xx > 0:
                self.XXStress['totalTens'] += xx
                self.XXStress['beamTens'] +=1
                if self.XXStress['maxTens'] < xx:
                    self.XXStress['maxTens'] = xx 
                if xx > self.material['allowedXXTension']:
                    self.XXStress['failTens'] +=1
            else:
                xx = abs(xx)
                self.XXStress['totalComp'] += xx
                self.XXStress['beamComp'] += 1
                if self.XXStress['maxComp'] < xx:
                    self.XXStress['maxComp'] = xx
                if xx > self.material['allowedXXCompression']:
                    self.XXStress['failComp'] +=1

            #calculate stresses on XY plane
            if xy > 0:
                self.XYStress['totalTens'] += xy
                self.XYStress['beamTens'] +=1
                if self.XYStress['maxTens'] < xy:
                    self.XYStress['maxTens'] = xy 
                if xy > self.material['allowedXYTension']:
                    self.XYStress['failTens'] +=1
            else:
                xy = abs(xy)
                self.XYStress['totalComp'] += xy
                self.XYStress['beamComp'] += 1
                if self.XYStress['maxComp'] < xy:
                    self.XYStress['maxComp'] = xy
                if xy > self.material['allowedXYCompression']:
                    self.XYStress['failComp'] +=1

            #calculate stresses on ZX plane        
            if zx > 0:
                self.ZXStress['totalTens'] += zx
                self.ZXStress['beamTens'] +=1
                if self.ZXStress['maxTens'] < zx:
                    self.ZXStress['maxTens'] = zx 
                if zx > self.material['allowedZXTension']:
                    self.ZXStress['failTens'] +=1
            else:
                zx = abs(zx)
                self.ZXStress['totalComp'] += zx
                self.ZXStress['beamComp'] += 1
                if self.ZXStress['maxComp'] < zx:
                    self.ZXStress['maxComp'] = zx
                if zx > self.material['allowedZXCompression']:
                    self.ZXStress['failComp'] +=1
        #handy for debugging
        #self.print_stresses()
                    

        self.XXStress['avrTens'] = self.XXStress['totalTens'] / self.XXStress['beamTens'] 
        self.XYStress['avrTens'] = self.XYStress['totalTens'] / self.XYStress['beamTens'] 
        self.ZXStress['avrTens'] = self.ZXStress['totalTens'] / self.ZXStress['beamTens']  
        self.XXStress['avrComp'] = self.XXStress['totalComp'] / self.XXStress['beamComp'] 
        self.XYStress['avrComp'] = self.XYStress['totalComp'] / self.XYStress['beamComp'] 
        self.ZXStress['avrComp'] = self.ZXStress['totalComp'] / self.ZXStress['beamComp'] 

    def print_stresses(self):
        print "XX Stress: ",str(self.XXStress)
        print "XY Stress: ",str(self.XYStress)
        print "ZX Stress: ",str(self.ZXStress)

    def test_mesh(self):
        self.create_graph()
        self.apply_stresses()
        self.create_slf_file()
        self.test_slf_file()
	if showAnalysis == True:
            self.show_analysis()
        self.parse_results()

    def show_mesh(self):
        self.create_graph()
        self.apply_stresses()
        self.create_slf_file()
        self.test_slf_file()
        self.parse_results()
        print "load bearing nodes:"
        print str(self.loadNodes)
        self.print_stresses()
        self.show_analysis()

        

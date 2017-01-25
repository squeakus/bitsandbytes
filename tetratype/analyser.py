import subprocess

# global class variables
SHOW = False
DEFAULT_FIT = 1000000

def eval_or_exec(program):
    """An unpleasant limitation in Python is the distinction between
    eval and exec. The former can only be used to return the value
    of a simple expression (not a statement) and the latter does not
    return anything."""

    program = python_filter(program)
    try:
        retval = eval(program)
    except SyntaxError:
        XXXeval_or_exec_outputXXX = None
        exec(program)
        retval = XXXeval_or_exec_outputXXX
    return retval

def python_filter(txt):
    """Create correct python syntax. We use {} to track indentation,
    which is not ideal because of the clash with dict literals.
    """
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

def log_error(phenotype, msg):
    log_file = open('err.log', 'a')
    log_file.write(msg+'\n')
    log_file.write(phenotype+'\n')
    log_file.close()


#this class will create the meshes and analyise them
class Analyser():
    broken = 0
    def __init__(self, unique_id, program, save=False):
        self.name = 'testMesh'
        self.unique_id = unique_id
        #stores all the slffea values
        self.my_graph = None
        self.save_mesh = save
        self.program = program
        self.zload_total = 10
        self.yload_total = 0
        self.zload_avr = 0
        self.zload_avr = 0
        self.max_comp = 1000000
        self.max_tens = 300000        
        self.edge_list = []
        self.node_list = []
        self.fixed_list = []
        self.load_nodes = []
        self.load_elems = []
        self.max = {'x':0, 'y':0, 'z':0}
        self.min = {'x':0, 'y':0, 'z':0}

        self.xx_stress = {'totalTens': 1, 'totalComp': 1,
                         'avrTens': 0, 'avrComp': 0,
                         'maxTens': 0, 'maxComp': 0,
                         'beamTens': 1, 'beamComp': 1,
                         'failTens': 0, 'failComp': 0}
        self.xy_stress = {'totalTens': 1, 'totalComp': 1,
                         'avrTens': 0, 'avrComp': 0,
                         'maxTens': 0, 'maxComp': 0,
                         'beamTens': 1, 'beamComp': 1,
                         'failTens': 0, 'failComp': 0}
        self.zx_stress = {'totalTens': 1, 'totalComp': 1,
                         'avrTens': 0, 'avrComp': 0,
                         'maxTens': 0, 'maxComp': 0,
                         'beamTens': 1, 'beamComp': 1,
                         'failTens': 0, 'failComp': 0}
        self.material = {}

        self.material['width'] = 0.1 #originally 100
        self.material['height'] = 0.2 #originally 200
        self.material['emod'] = 10000000000 #N/m2
        self.material['density'] = 530 #originally 5300
        self.material['unitweight'] = ((float(self.material['width']))
                                       * (float(self.material['height']))
                                       * float(self.material['density']))
        self.material['area'] = ((float(self.material['width']))
                                 * (float(self.material['height'])))
        self.material['iz'] = (((float(self.material['width']))
                                * (float(self.material['height']) ** 3))
                                / 12)
        self.material['iy'] = (((float(self.material['height']))
                                * (float(self.material['width']) ** 3))
                                / 12)
        self.material['xx_tension'] = (1.8 * 10 ** 7)
        self.material['xx_compression'] = (2.3 * 10 ** 7)
        self.material['xy_tension'] = (0.6 * 10 ** 6)
        self.material['xy_compression'] = (8.4 * 10 ** 6)
        self.material['zx_tension'] = self.material['xy_tension']
        self.material['zx_compression'] = self.material['xy_compression']
        self.material['maxmoment'] = (3 * 10 ** 7)
        
    def create_graph(self):
        self.my_graph = eval_or_exec(self.program)
        self.parse_graph(self.my_graph)
        
    def parse_graph(self, my_graph):
        for node in my_graph.nodes():
            #xyz =my_graph.get_node_data(node)
            xyz = my_graph.node[node]['xyz']
            label = my_graph.node[node]['label']
            node = {'id':str(node), 'x':xyz[0], 'y':xyz[1],
                    'z':xyz[2],'label':label}
            self.node_list.append(node)
        for idx, edge in enumerate(my_graph.edges_iter()):
            edge = {'id':str(idx), 'pt_a':str(edge[0]), 'pt_b':str(edge[1])}
            self.edge_list.append(edge)   
        if self.save_mesh == True:
            self.create_mesh()
        
    def create_mesh(self, name = 'indiv'):
        if name == 'indiv':
            filename = "./indiv."+str(self.unique_id)+".mesh"
        else:
            filename = name+'.mesh' 
        mesh = open(filename,'w')
        mesh.write("MeshVersionFormatted 1\nDimension\n3 \n")
        mesh.write("Vertices\n"+str(len(self.node_list))+" \n")
        for node in self.node_list:
            mesh.write(str(node['x'])+" "+str(node['y'])+" "
                       +str(node['z'])+" 0  \n")
        mesh.write("Edges\n"+str(len(self.edge_list))+" \n")
        for edge in self.edge_list:
            pt_a, pt_b = int(edge['pt_a']), int(edge['pt_b'])
            #pt_a, pt_b = edge['pt_a'],edge['pt_b']
            mesh.write(str(pt_a+1)+" "+str(pt_b+1)+" 0 \n")
            #mesh.write(str(pt_a)+" "+str(pt_b)+" 0 \n")
        mesh.write("End\n")
        mesh.close()

    def create_debug(self):
        mesh = open("debug.mesh",'w')
        for node in self.node_list:
            mesh.write('node '+node['id']+' '+str(node['x'])
                       +' '+str(node['y'])+' '+str(node['z'])
                       +' '+node['label']+'\n')
        for edge in self.edge_list:
            mesh.write('edge '+edge['id']+' '+edge['pt_a']
                       +' '+edge['pt_b']+'\n')

    def apply_stresses(self):        
        #build fixed points list and load list
        for node in self.node_list:
            x, y, z = node['x'], node['y'], node['z']
            if node['label'] == 'fixed':
                self.fixed_list.append([node, True])
            if node['label'] == 'load':
                self.load_nodes.append(int(node['id']))
        #print "Fixed:",len(self.fixed_list), "Loaded",len(self.load_nodes)
        #SLFFEA applies load to edges, find edges connecting load_nodes 
        for edge in self.edge_list:
            id_a, id_b = int(edge['pt_a']), int(edge['pt_b'])
            if id_a in self.load_nodes or id_b in self.load_nodes:
                # pt_a, pt_b = None, None
                # for node in self.node_list:
                #     if int(node['id']) == id_a:
                #         pt_a = node
                #     elif int(node['id']) == id_b:
                #         pt_b = node    
                # if pt_a['y'] == pt_b['y']:
                self.load_elems.append(edge['id'])

        # if len(self.load_elems)>0:
        #     self.zload_avr = self.zload_total/len(self.load_elems)
        #     print "load",self.zload_avr
        # else:
        #     print "no loadNodes"
        self.zload_avr = 1

    def create_slf_file(self):
        mesh = open(self.name, 'w')
        mesh.write('numel numnp nmat nmode (This is for a beam bridge)\n')
        mesh.write(str(len(self.edge_list))+' '+str(len(self.node_list))
                   +' 1 0\n')
        mesh.write('matl no., E mod, Poiss. Ratio,density, Area, Iy, Iz\n')
        mesh.write('0 '+str(self.material['emod'])+' 0.0000 '
                   +str(self.material['density'])+' '
                   +str(self.material['area'])+' '
                   +str(self.material['iy'])+' '
                   +str(self.material['iz']) + '\n')
        mesh.write('el no.,connectivity, matl no, element type\n')
        for i, edge in enumerate(self.edge_list): 
            mesh.write(str(i)+' '+str(edge['pt_a'])+' '
                       +str(edge['pt_b'])+' 0 2 \n')
        mesh.write('node no., coordinates\n')
        for node in self.node_list:
            mesh.write(node['id']+' '+str(node['x'])+' '
                       +str(node['y'])+' '+str(node['z'])+"\n")
        mesh.write("element with specified local z axis: x, y, z component\n -10\n")
        mesh.write('prescribed displacement x: node  disp value\n')
        for node in self.fixed_list:
            if node[1] == True:
                mesh.write(node[0]['id']+" 0.0\n")
        mesh.write('-10\nprescribed displacement y: node  disp value\n')
        for node in self.fixed_list:
            mesh.write(node[0]['id']+" 0.0\n")
        mesh.write('-10\nprescribed displacement z: node  disp value\n')
        for node in self.fixed_list:
            mesh.write(node[0]['id']+" 0.0\n")
        mesh.write('-10\nprescribed angle phi x: node angle value\n -10')
        mesh.write('\nprescribed angle phi y: node angle value\n -10')
        mesh.write('\nprescribed angle phi z: node angle value\n -10')
        mesh.write('\nnode with point load x, y, z and 3 moments phi x, phi y, phi z\n-10\n')
        mesh.write('element with distributed load in local beam y and z coordinates\n')
        for elem in self.load_elems: 
            mesh.write(elem+' -0 -'+str(self.zload_avr)+' \n') 
        mesh.write('-10\nelement no. and gauss pt. no. with local stress vector xx and moment xx,yy,zz\n-10')
        mesh.close()

    def test_slf_file(self):
        #run the structural analysis software
        cmd = 'bm'
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, 
                                   stdin=subprocess.PIPE)
        process.communicate(self.name)

    def show_analysis(self):
        cmd = "echo "+self.name+'.obm | bmpost'
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                                   stdin=subprocess.PIPE)
        process.communicate(self.name)

    def parse_results(self):
        stressHeader1 = "element no. and nodal pt. no. with local stress xx,xy,zx and moment xx,yy,zz"
        stressHeader2 = "element no. and gauss pt. no. with local stress xx,xy,zx and moment xx,yy,zz"
        stress_list = []
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
                        log_error("structural failure", self.program)
                        print "**************STRUCTURE FAILED****************"
                        xx_float, xy_float, zx_float = DEFAULT_FIT, DEFAULT_FIT, DEFAULT_FIT
                        xx,xy,zx = int(xx_float), int(xy_float), int(zx_float)
                        stress = {'xx':xx,'xy':xy,'zx':zx}
                        stress_list.append(stress)
                        exit()
                        break
                    else:  
                        xx_float, xy_float, zx_float = float(result[2]), float(result[3]), float(result[4])
                        xx, xy, zx = int(xx_float), int(xy_float), int(zx_float)
                        stress = {'xx':xx, 'xy':xy, 'zx':zx}
                        stress_list.append(stress)
                        line = lines.next()
        results.close()
        self.calculate_stresses(stress_list)
    
    def calculate_stresses(self, stress_list):
        for stress in stress_list:
            xx, xy, zx = stress['xx'], stress['xy'], stress['zx']
            #calculate stresses on XX plane
            df = DEFAULT_FIT
            if xx == df or xy == df or zx == df:
                self.xx_stress = {'totalTens':df, 'totalComp':df, 
                                  'maxTens':df, 'maxComp':df, 
                                  'beamTens':1, 'beamComp':1,
                                  'failTens':df, 'failComp':df}
                self.xy_stress = {'totalTens':df, 'totalComp':df,
                                  'maxTens':df, 'maxComp':df, 
                                  'beamTens':1, 'beamComp':1, 
                                  'failTens':df, 'failComp':df}
                self.zx_stress = {'totalTens':df, 'totalComp':df,
                                  'maxTens':df, 'maxComp':df,
                                  'beamTens':1, 'beamComp':1,
                                  'failTens':df,'failComp':df}
            if xx > 0:
                self.xx_stress['totalTens'] += xx
                self.xx_stress['beamTens'] += 1
                if self.xx_stress['maxTens'] < xx:
                    self.xx_stress['maxTens'] = xx 
                if xx > self.material['xx_tension']:
                    self.xx_stress['failTens'] += 1
            else:
                xx = abs(xx)
                self.xx_stress['totalComp'] += xx
                self.xx_stress['beamComp'] += 1
                if self.xx_stress['maxComp'] < xx:
                    self.xx_stress['maxComp'] = xx
                if xx > self.material['xx_compression']:
                    self.xx_stress['failComp'] += 1

            #calculate stresses on XY plane
            if xy > 0:
                self.xy_stress['totalTens'] += xy
                self.xy_stress['beamTens'] += 1
                if self.xy_stress['maxTens'] < xy:
                    self.xy_stress['maxTens'] = xy 
                if xy > self.material['xy_tension']:
                    self.xy_stress['failTens'] += 1
            else:
                xy = abs(xy)
                self.xy_stress['totalComp'] += xy
                self.xy_stress['beamComp'] += 1
                if self.xy_stress['maxComp'] < xy:
                    self.xy_stress['maxComp'] = xy
                if xy > self.material['xy_compression']:
                    self.xy_stress['failComp'] += 1

            #calculate stresses on ZX plane        
            if zx > 0:
                self.zx_stress['totalTens'] += zx
                self.zx_stress['beamTens'] += 1
                if self.zx_stress['maxTens'] < zx:
                    self.zx_stress['maxTens'] = zx 
                if zx > self.material['zx_tension']:
                    self.zx_stress['failTens'] += 1
            else:
                zx = abs(zx)
                self.zx_stress['totalComp'] += zx
                self.zx_stress['beamComp'] += 1
                if self.zx_stress['maxComp'] < zx:
                    self.zx_stress['maxComp'] = zx
                if zx > self.material['zx_compression']:
                    self.zx_stress['failComp'] += 1

        self.xx_stress['avrTens'] = self.xx_stress['totalTens'] / self.xx_stress['beamTens'] 
        self.xy_stress['avrTens'] = self.xy_stress['totalTens'] / self.xy_stress['beamTens'] 
        self.zx_stress['avrTens'] = self.zx_stress['totalTens'] / self.zx_stress['beamTens']  
        self.xx_stress['avrComp'] = self.xx_stress['totalComp'] / self.xx_stress['beamComp'] 
        self.xy_stress['avrComp'] = self.xy_stress['totalComp'] / self.xy_stress['beamComp'] 
        self.zx_stress['avrComp'] = self.zx_stress['totalComp'] / self.zx_stress['beamComp'] 

    def calculate_fitness(self): 
        total_length = (0)
        for edge in self.edge_list:
            n1, n2 = edge['pt_a'], edge['pt_b']
            index = edge['id']
            for node in self.node_list:
                if node['id'] == n1:
                    n1x,n1y,n1z = node['x'],node['y'],node['z']
                if node['id'] == n2:
                    n2x,n2y,n2z = node['x'],node['y'],node['z']
            length = ((((n1x-n2x)**2)+((n1y-n2y)**2)+((n1z-n2z)**2))**(1/2)) 
            total_length = float(total_length) + float(length)              
        weight = float(total_length) * self.material['unitweight']

        x,y,z = self.xx_stress['failTens'], self.xy_stress['failTens'],self.zx_stress['failTens'] 
        total_failed = x + y + z
        avr_comp = self.xx_stress['avrComp']+self.xy_stress['avrComp']+self.zx_stress['avrComp']
        avr_tens = self.xx_stress['avrTens']+self.xy_stress['avrTens']+self.zx_stress['avrTens']
        norm_comp = self.normalize(avr_comp,[1,1000], [0,self.max_comp])
        norm_tens = self.normalize(avr_tens,[1,1000], [0,self.max_tens])
        fitness = norm_comp + norm_tens
        #print "fitness:", fitness, "weight:", weight
        return fitness, weight

    def normalize(self,value,newRange,oldRange):
        if value > oldRange[1]:
            value = oldRange[1]
        normalized = (newRange[0]+(value-oldRange[0])*(newRange[1]-newRange[0]))/oldRange[1]-oldRange[0] 
        return normalized



    def print_stresses(self):
        print "XX Stress: ", str(self.xx_stress)
        print "XY Stress: ", str(self.xy_stress)
        print "ZX Stress: ", str(self.zx_stress)

    def test_mesh(self):
        self.create_graph()
        self.apply_stresses()
        self.create_slf_file()
        self.test_slf_file()
        if SHOW == True:
            self.show_analysis()
        self.parse_results()

    def show_mesh(self):
        self.create_graph()
        self.apply_stresses()
        self.create_slf_file()
        self.test_slf_file()
        self.parse_results()
        print "load bearing nodes:"
        print str(self.load_nodes)
        self.print_stresses()
        self.show_analysis()


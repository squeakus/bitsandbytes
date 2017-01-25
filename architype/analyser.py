""" This class contains functions for parsing the programs generated
by the grammar, creating the resulting graphs, converting them to
slffea files, running slffea, collating the results and using them to
generate a fitness value. 

ALL UNITS IN MILLIMETERS, NEWTONS
  
Copyright (c) 2010
Jonathan Byrne, Michael Fenton, Erik Hemberg and James McDermott
Hereby licensed under the GNU GPL v3."""
import subprocess, time, operator, graph, random
from geometry import *
from constraints import *
from math import exp, sin, cos, pi, sqrt
from operator import itemgetter
import architype as ARC
import optimizer as OPT

# global class variables
SHOW_ANALYSIS = False
OPTIMIZE = False # Turns on the pre-fitness function optimizer which will optimize the whole population
DEFAULT_FIT = 100000000000000 
#random.seed(0)

def eval_or_exec(s):
    """Handles different return vals from eval/exec"""
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


def python_filter(txt):
    """Converts text into indented python code"""
    counter = 0
    if txt == None:
        log_error("None", "no program generated")
        return 0
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
    """any problems dumped to err.log"""
    print "logging error(" + str(time.clock) + "):", msg
    errFile = open('err.log', 'a')
    errFile.write("error(" + str(time.clock) + "):" + msg + "\n")
    errFile.write(str(phenotype) + "\n")
    errFile.close()


def lt(val_a, val_b):
    """less than op for conditionals in generated code"""
    return operator.lt(val_a, val_b)


def le(val_a, val_b):
    """less than or equal op for conditionals in generated code"""
    return operator.le(val_a, val_b)


def gt(val_a, val_b):
    """greater than op for conditionals in generated code"""
    return operator.gt(val_a, val_b)


class Analyser():
    """this class reads a standard mesh and generates an slffea mesh.
    It is then analysed by slffea and the result is processed to
    generate a fitness value"""

    def __init__(self, unique_id, program, save=False):
        self.name = 'testMesh'
        self.unique_id = unique_id
        #stores all the slffea values
        self.max_comp = 9000000
        self.max_tens = 3000000
        self.my_graph = None
        self.save_mesh = save
        self.program = program
        self.BROKEN = False
        self.GROUND_BROKEN = False # Change this to true to test the case with a broken grounding wire
        self.load_case = "Case5b" # Choose from "Case1a" (extreme wind), "Case2a" (Uniform, heavy ice), "Case3" (Combined wind and ice), "Case5b" (Security loads, Broken wire condition)
        self.terrain_category = 2 
        self.compliance = 1000000000000
        self.max_displacement = 3000
        self.beams=[]
        self.edge_list = []
        self.node_list = []
        self.new_node_list = []
        self.fixed_list = []
        self.load_nodes = []
        self.ground_node = []
        self.break_node = []
        self.load_elems = []
        self.nodeselfloads=[]
        self.beam_comp_allows=[]
        self.material = {}
        self.material['name'] = 'steel'
        
        if self.material['name'] == 'steel':
            self.material['number'] = 60
            self.material['leg_number'] = 120
            self.material['allowed_xx_compression'] = self.material['allowed_xx_tension'] = self.material['allowed_xy_tension'] = self.material['allowed_xy_compression'] = self.material['allowed_zx_tension'] = self.material['allowed_zx_compression'] = (275) # 275 N/mm2
            self.material['maxmoment'] = (3*10**7)
        
        if self.material['name'] == 'timber':
            self.material['width'] = 100 # mm
            self.material['height'] = 200 # mm
            self.material['emod'] = 10000 #N/mm2
            self.material['density'] = 5.3 * 10**(-6) #originally 5300
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
            self.material['allowed_xx_tension'] = (1.8 * 10 ** 7)
            self.material['allowed_xx_compression'] = (2.3 * 10 ** 7)
            self.material['allowed_xy_tension'] = (0.6 * 10 ** 6)
            self.material['allowed_xy_compression'] = (8.4 * 10 ** 6)
            self.material['allowed_zx_tension'] = self.material['xy_tension']
            self.material['allowed_zx_compression'] = self.material['xy_compression']
            self.material['maxmoment'] = (3 * 10 ** 7)

    def create_graph(self):
        """execute program to create the graph """
        self.my_graph = eval_or_exec(self.program)
        self.parse_graph()

    def parse_graph(self):
        self.assign_size()
        """gather nodes and edges from output graph"""
        for node in self.my_graph.nodes():
            xyz = self.my_graph.get_node_data(node)
            label = self.my_graph.node[node]['label']
            node = {'id': str(node), 'x': xyz[0], 'y': xyz[1],
                    'z': xyz[2], 'label': label}
            self.node_list.append(node)
        for idx, edge in enumerate(self.my_graph.edges_iter()):
            n1 = str(edge[0])
            n2 = str(edge[1])
            index = str(idx)
            for node in self.node_list:
                if node['id'] == n1:
                    node1 = node
                if node['id'] == n2:
                    node2 = node
            if (node1['label'] == 'crossbrace' and node2['label'] == 'crossbrace') or (node1['label'] == 'leg' and node2['label'] == 'crossbrace') or (node1['label'] == 'crossbrace' and node2['label'] == 'leg') or (node1['label'] == 'leg' and node2['label'] == 'leg') or (node1['label'] == 'leg' and node2['label'] == 'base') or (node1['label'] == 'base' and node2['label'] == 'leg'):
                length = three_d_line_length(node1,node2) # in millimeters
                mass = length * float(self.beams[self.material['leg_number']]['unitweight']) # answer is in Newtons
                area = self.beams[self.material['leg_number']]['area'] # in millimeters squared
                name = 'legs'
                edge = {'id':str(idx),'pt_a':str(edge[0]),'pt_b':str(edge[1]),
                        'material':str(self.material['leg_number']),'length':str(length),
                        'mass':str(mass),'area':str(area),'label':name}
                self.edge_list.append(edge)
            else:
                length = three_d_line_length(node1,node2) # in millimeters
                mass = length * float(self.beams[self.material['number']]['unitweight']) # answer is in Newtons
                area = self.beams[self.material['number']]['area'] # in millimeters squared
                name = 'crossbrace'
                edge = {'id':str(idx),'pt_a':str(edge[0]),'pt_b':str(edge[1]),
                        'material':str(self.material['number']),'length':str(length),
                        'mass':str(mass),'area':str(area),'label':name}
                self.edge_list.append(edge)
        if self.save_mesh:
            self.create_mesh()

    def create_mesh(self, name='indiv'):
        """produces mesh for slffea"""
        if name == 'indiv':
            filename = "population/indiv." + str(self.unique_id) + ".mesh"
        else:
            filename = name + '.mesh'
        mesh = open(filename, 'w')
        mesh.write("MeshVersionFormatted 1\nDimension\n3 \n")
        mesh.write("Vertices\n" + str(len(self.node_list)) + " \n")
        for node in self.node_list:
            mesh.write(str(node['x']) + " " + str(node['y'])
                       + " " + str(node['z']) + " 0  \n")
        mesh.write("Edges\n" + str(len(self.edge_list)) + " \n")
        for edge in self.edge_list:
            pt_a, pt_b = int(edge['pt_a']), int(edge['pt_b'])
            mesh.write(str(pt_a + 1) + " " + str(pt_b + 1) + " 0 \n")
        mesh.write("End\n")
        mesh.close()

    def save_dxf(self, gen, name):
        """outputs nodes and edges in dxf format for other software"""
        if name == 'indiv':
            filename = "dxf/gen" + str(gen) + "ind"  + str(self.unique_id) + ".dxf"       
        DXF = file(filename, 'w')
        DXF.write('  0\n')
        DXF.write('SECTION\n')
        DXF.write('  2\n')
        DXF.write('ENTITIES\n')
        for edge in self.edge_list:
            for node in self.node_list:
                if edge['pt_a'] == node['id']:
                    X1, Y1, Z1 = node['x'], node['y'], node['z']
                if edge['pt_b'] == node['id']:
                    X2, Y2, Z2 = node['x'], node['y'], node['z']
            DXF.write('  0\n')
            DXF.write('LINE\n')
            DXF.write('  8\n')
            DXF.write('Polygon\n')
            DXF.write(' 10\n')
            DXF.write(str(X1))
            DXF.write('\n 20\n')
            DXF.write(str(Y1))
            DXF.write('\n 30\n')
            DXF.write(str(Z1))
            DXF.write('\n 11\n')
            DXF.write(str(X2))
            DXF.write('\n 21\n')
            DXF.write(str(Y2))
            DXF.write('\n 31\n')
            DXF.write(str(Z2))
            DXF.write('\n')
        DXF.write('  0\n')
        DXF.write('ENDSEC\n')
        DXF.write('  0\n')
        DXF.write('EOF\n')
        DXF.close()

    def create_debug(self):
        """if mesh breaks, save a copy of it for analysis"""
        mesh = open("debug.mesh", 'w')
        for node in self.node_list:
            mesh.write('node ' + node['id'] + ' ' + str(node['x'])
                       + ' ' + str(node['y']) + ' ' + str(node['z'])
                       + ' ' + node['label'] + '\n')
        for edge in self.edge_list:
            mesh.write('edge ' + edge['id'] + ' ' + edge['pt_a']
                       + ' ' + edge['pt_b'] + '\n')

    def assign_wind_loads(self):
        """Computes the necessary wind loads for the structure
           and then assigns them to the appropriate nodes"""
        Vr = self.mean_hourly_wind_speed
        roa = 1.22 # kg/m3 - density of air in Great Britan
        if self.terrain_category == 2:
            # This calculates the variation of wind speed with height
            Kr = 1.10 # Terrain roughness factor
            Z0 = 0.01 # Terrain aerodynamic roughness parameter (meters)
            alpha = 0.14 # Power law index of variation of wind speed with height
            He = 0 # Effective height
            Cn = 1.2 # Overall drag/pressure coefficuent, dependent on solidarity ratio, educated guess
            Kcom = 1
            K0 = 1
            H = 55 # (meters) Presumably this is the height of the tower, it's very fucking vague
            K1 = (1+(alpha/2))*(10/H)**alpha
            S1 = (H/100.8)*((10/H)**alpha)
            K6 = max((H/10),10)
            S2 = (K6/100.8)*((10/H)**alpha)
            K2 = ((2/S1)+((2/(S2**2))*((e**(-S1))-1)))**0.5
            Gx = K1*K2*((3.976/Kr)-2.485)
            K3 = (1+(alpha/2))*((10/K6)**alpha)
            K4 = ((2/S2)+((2/(S2**2))*((e**(-S2))-1)))**0.5
            K5 = ((K6/H)**alpha)*(1-(1-(K6/H))**2)/(1-(1-(K6/H))**(alpha+2))
            Gy = K3*K4*K5*((3.976/Kr)-2.845)
            Gb = 0.9 # max(Gx,Gy) # this is the cheating method
            for node in self.node_list:
                if node['y']>0:
                    if (float(node['z'])/1000) >= 10+He:
                        Vz = Vr((((float(node['z'])/1000)-He)/10)**alpha)
                    elif (float(node['z'])/1000) < 10+He:
                        Vz = Vr(((0.25/(10+He))*(float(node['z'])/1000))+0.75)
                    Qz = (roa/2)*(Vz**2)
                    
                    
                    As = 0# Structural components of projected area on windward side - the area over which the wind pressure acts, to be broken into panels across the height of the structure.
                
                
                    PTW = Qz*As*Cn*(1+(Kcom*Gb))*K0 # PTW is the maximum wind load acting on a particular panel, it can be split in to 50% acting on the top and 50% on the bottom (or i presume the load can be spread evenly across all nodes within As)

    def assign_load_case(self):
        """All loads are in Newtons"""
        if self.load_case == "Case1a": # Extreme Wind
            self.mean_hourly_wind_speed = 35 # meters/sec
            self.BROKEN = False
            self.GROUND_BROKEN = False
            self.vertical_cable_load = 50000
            self.transverse_cable_load = 82000
            self.longitudinal_cable_load = 0
            self.vertical_ground_load = 5700
            self.transverse_ground_load = 15200
            self.longitudinal_ground_load = 0
        if self.load_case == "Case2a": # Uniform, heavy Ice
            self.BROKEN = False
            self.GROUND_BROKEN = False
            self.mean_hourly_wind_speed = 25 # meters/sec
            self.vertical_cable_load = 171000 
            self.transverse_cable_load = 0
            self.longitudinal_cable_load = 0
            self.vertical_ground_load = 46000
            self.transverse_ground_load = 0
            self.longitudinal_ground_load = 0
        if self.load_case == "Case3": # Combined wind & ice
            self.mean_hourly_wind_speed = 30 # meters/sec
            self.BROKEN = False
            self.GROUND_BROKEN = False
            self.vertical_cable_load = 86000
            self.transverse_cable_load = 70000
            self.longitudinal_cable_load = 0
            self.vertical_ground_load = 15300
            self.transverse_ground_load = 20900
            self.longitudinal_ground_load = 0
        if self.load_case == "Case5b": # Security loads. Broken wire condition
            self.BROKEN = True
            self.mean_hourly_wind_speed = 25 # meters/sec 
            self.vertical_cable_load = 50000
            self.vertical_cable_load_broken = 37500 # (to be placed on any single one of the cables, doesn't matter which)
            self.transverse_cable_load = 0
            self.longitudinal_cable_load = 99000 # (to be placed on any single one of the cables, the same one as above)
            self.vertical_ground_load = 5700
            self.vertical_ground_load_broken = 4200 # (to be placed on the grounding cable, during broken condition)
            self.transverse_ground_load = 0
            self.longitudinal_ground_load = 16000 # (to be placed on the grounding cable, during broken condition)
        if self.load_case == "Test": # Security loads. Broken wire condition
            self.BROKEN = True
            self.mean_hourly_wind_speed = 35 # meters/sec 
            self.vertical_cable_load = 171000
            self.vertical_cable_load_broken = 37500 # (to be placed on any single one of the cables, doesn't matter which)
            self.transverse_cable_load = 82000
            self.longitudinal_cable_load = 99000 # (to be placed on any single one of the cables, the same one as above)
            self.vertical_ground_load = 46000
            self.vertical_ground_load_broken = 4200 # (to be placed on the grounding cable, during broken condition)
            self.transverse_ground_load = 15200
            self.longitudinal_ground_load = 16000 # (to be placed on the grounding cable, during broken condition)


    def apply_stresses(self):
        """build fixed points list and load list"""
        for node in self.node_list:
            if  node['label'] == 'base':
                self.fixed_list.append([node, True])            
            if node['label'] == 'line':
                self.load_nodes.append(int(node['id']))
            if node['label'] == 'ground':
                self.ground_node.append(int(node['id']))
        if self.BROKEN:
            point = self.load_nodes.pop(-2)
            self.break_node.append(point) 
            if self.GROUND_BROKEN == False:
                joint = self.load_nodes.pop()
                self.break_node.append(joint)          

        #SLFFEA applies load to edges, find edges connecting load_nodes
        for edge in self.edge_list:
            pt_a, pt_b = int(edge['pt_a']), int(edge['pt_b'])
            if pt_a in self.load_nodes or pt_b in self.load_nodes:
                self.load_elems.append(edge['id'])         
            
        #SLFFEA doesn't consider the mass of the element; we have to compute this ourselves and
        #add it as a point load to the nodes at each end of the element
            load = float(edge['mass']) / 2   # load per node is in newtons                           
            loadA = [pt_a,load]
            loadB = [pt_b,load]
            self.nodeselfloads.append(loadA)
            self.nodeselfloads.append(loadB)
        self.nodeselfloads.sort(key=itemgetter(0))
        #Now we need to find the nodes with the same node[0] (i.e. the same nodes) and add up the individual loads (node[1]) to return the total load on that particular node.        

        if self.nodeselfloads:
            while len(self.nodeselfloads) > (len(self.node_list) + 20):  #
                last = self.nodeselfloads[-1]
                for i in range(len(self.node_list)):
                    if i < len(self.nodeselfloads):
                        if last[0] == self.nodeselfloads[i][0]:  
                            last[1] = last[1] + self.nodeselfloads[i][1]
                            del self.nodeselfloads[i]
                        else:
                            last = self.nodeselfloads[i]
            while len(self.nodeselfloads) > (len(self.node_list)):            
                last = self.nodeselfloads[-1]
                for i in range(len(self.node_list)):
                    if last[0] == self.nodeselfloads[-i][0]:
                        last[1] = last[1] + self.nodeselfloads[-i][1]
                        del self.nodeselfloads[-i]
                    else:
                        last = self.nodeselfloads[-i]
        else:
            print "ERROR!!********************NO NODESELFLOADS THING!************************"           
           
    def assign_size(self):
        allowables = open('./tables/CHS_compression_resistance_tables.txt', 'r')
        tables = open('./tables/CHSTables.txt', 'r')
        number = 0
        for i, line in enumerate(allowables):
            if line.startswith('#'):
                number = number + 1
                allowables.readlines
            else:
                line = line.split()
                idx = i-number
                one, one_point_five, two, two_point_five, three, three_point_five, four, five, six, seven, eight, nine, ten = float(line[3]),float(line[4]),float(line[5]),float(line[6]),float(line[7]),float(line[8]),float(line[9]),float(line[10]),float(line[11]),float(line[12]),float(line[13]),float(line[14]),float(line[15])               
                material_properties = {'id':idx,'1':one,'1.5':one_point_five,'2':two,'2.5':two_point_five,
                                        '3':three,'3.5':three_point_five,'4':four,'5':five,'6':six,'7':seven,
                                        '8':eight,'9':nine,'10':ten}
                self.beam_comp_allows.append(material_properties)
        number = 0
        for i, line in enumerate(tables):
            if line.startswith('#'):
                number = number + 1
                tables.readlines
            else:
                line = line.split()
                idx = i-number
                size = line[0]
                unitweight = float(line[2])/100 # in newtons per millimeter
                area = float(line[3]) * 100 # in millimeters squared
                ix = iy = float(line[5])*10000 # in millimeters to the four
                emod = 210000 # in Newtons/millimeters or MegaPascals
                density = 7.85 * 10 ** (-6)  # in kg/millimeter cubed
                material_properties = {'id':idx,'size':size,'unitweight':unitweight,
                                       'area':area,'ix':ix,'iy':iy,'emod':emod,
                                       'density':density}
                self.beams.append(material_properties)

    def create_slf_file(self):
        """outputs an slf file in beam format"""
        mesh = open(self.name, 'w') 
        mesh.write('numel numnp nmat nmode (This is for a beam bridge)\n')
        mesh.write(str(len(self.edge_list))+'\t'+str(len(self.node_list))
                    + '\t'+str(len(self.beams)) + '\t0\n')
        mesh.write('matl no., E mod, Poiss. Ratio,density, Area, Iy, Iz\n')
        tables = open('./tables/CHSTables.txt', 'r')
        for i,beam in enumerate(self.beams):
             mesh.write(str(i)+' '+str(self.beams[i]['emod'])+'\t0.3000\t'
                        + str(self.beams[i]['density'])+'\t'+str(self.beams[i]['area'])
                        + '\t'+str(self.beams[i]['iy'])+'\t'+str(self.beams[i]['ix']) + '\n')           
        mesh.write('el no.,connectivity, matl no, element type\n')
        for i, edge in enumerate(self.edge_list): 
            mesh.write(str(i)+'\t'+str(edge['pt_a'])+'\t'+str(edge['pt_b'])
                       + '\t'+str(edge['material'])+'\t2 \n')
        mesh.write('node no., coordinates\n')
        for node in self.node_list:
            mesh.write(node['id']+'\t'+str(node['x'])+'\t'+str(node['y'])+'\t'+str(node['z'])+"\n")
        mesh.write("element with specified local z axis: x, y, z component\n -10\n")
        mesh.write('prescribed displacement x: node  disp value\n')
        for node in self.fixed_list:
#            if node[1] == True: # un-comment when dealing with fixed-roller structures
            mesh.write(node[0]['id']+"\t0.0\n")
        mesh.write('-10\nprescribed displacement y: node  disp value\n')
        for node in self.fixed_list:
            mesh.write(node[0]['id']+"\t0.0\n")
        mesh.write('-10\nprescribed displacement z: node  disp value\n')
        for node in self.fixed_list:
            mesh.write(node[0]['id']+"\t0.0\n")
        mesh.write('-10\nprescribed angle phi x: node angle value\n')
        for node in self.fixed_list:
#            if node[1] == True: # un-comment when dealing with fixed-roller structures
            mesh.write(node[0]['id']+"\t0.0\n")
        mesh.write('-10\nprescribed angle phi y: node angle value\n')
        for node in self.fixed_list:
            mesh.write(node[0]['id']+"\t0.0\n")
        mesh.write('-10\nprescribed angle phi z: node angle value\n')
        for node in self.fixed_list:
            mesh.write(node[0]['id']+"\t0.0\n")
        mesh.write('-10\nnode with point load x, y, z and 3 moments phi x, phi y, phi z\n')          
        if self.BROKEN:
            for node in self.nodeselfloads: 
                trans = 0
                broken_long = 0
                for thing in self.load_nodes:
                    if thing == node[0]:
                        node[1] = node[1] + self.vertical_cable_load
                        trans = self.transverse_cable_load 
                if self.GROUND_BROKEN:
                    for thing in self.ground_node:
                        if thing == node[0]:
                            node[1] = node[1] + self.vertical_ground_load_broken
                            trans = self.transverse_ground_load
                            broken_long = self.longitudinal_ground_load
                    for thing in self.break_node:
                        if thing == node[0]:
                            node[1] = node[1] + self.vertical_cable_load_broken
                            broken_long = self.longitudinal_cable_load
                            trans = self.transverse_cable_load
                else:
                    for thing in self.ground_node:
                        if thing == node[0]:
                            node[1] = node[1] + self.vertical_ground_load
                            trans = self.transverse_ground_load
                    for thing in self.break_node:
                        if thing == node[0]:
                            node[1] = node[1] + self.vertical_cable_load_broken
                            broken_long = self.longitudinal_cable_load 
                            trans = self.transverse_cable_load
                mesh.write(str(node[0])+'\t'+str(broken_long)+'\t'+str(trans)+'\t-'+str(round(node[1],5))+'\t0\t0\t0\n')
        else:
            for node in self.nodeselfloads: 
                trans = 0
                for yolk in self.load_nodes:
                    if yolk == node[0]:
                        node[1] = node[1] + self.vertical_cable_load
                        trans = self.transverse_cable_load
                for thong in self.ground_node:
                    if thong == node[0]:
                        node[1] = node[1] + self.vertical_ground_load
                        trans = self.transverse_ground_load
                mesh.write(str(node[0])+'\t0\t'+str(trans)+'\t-'+str(round(node[1],5))+'\t0\t0\t0\n')
        mesh.write('-10\nelement with distributed load in global beam y and z coordinates\n')                    
        mesh.write('-10\nelement no. and gauss pt. no. with local stress vector xx and moment xx,yy,zz\n-10')
        mesh.close()

    def test_slf_file(self):
        """run the structural analysis software, you must have
        compiled slffea and added bm and bmpost to /usr/local/bin"""
        cmd = 'bm'
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                                   stdin=subprocess.PIPE)
        result = process.communicate(self.name)
        result
        output = open(self.name +'.output', 'w')
        output.write(str(result))
        output.close()
        answer = open(self.name +'.output', 'r')
        for line in answer:
            thing = line.strip()
            yes = thing.split()
            self.compliance = yes[-3]
            
    def show_analysis(self):
        """use bmpost to show stresses"""
        cmd = "echo " + self.name + '.obm | bmpost'
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                                   stdin=subprocess.PIPE)
        process.communicate(self.name)

    def parse_results(self):
        """ read the results of slffea output, check if it calculated
        the results correctly"""
        header1 = ("element no. and nodal pt. no. with local stress"
                   + " xx,xy,zx and moment xx,yy,zz")
        header2 = ("element no. and gauss pt. no. with local stress"
                   + "xx,xy,zx and moment xx,yy,zz")
        header3 = ("prescribed displacement x: node  disp value")
        header4 = ("prescribed displacement y: node  disp value")
        header5 = ("prescribed displacement z: node  disp value")
        header6 = ("node no., coordinates")
        stress_list = []
        results = open(self.name + '.obm', 'r')  # opens the results file
        lines = iter(results)
        for line in lines:
            if line.startswith(header6):
                line = lines.next()
                while line[0] != ('element'):
                    result = line.split()
                    if result[0] == ('element'):
                        break
                    idx, x, y, z = str(result[0]),float(result[1]),float(result[2]),float(result[3])
                    node = {'id': idx, 'x': x, 'y': y, 'z': z}
                    self.new_node_list.append(node)
                    line = lines.next()
            if line.startswith(header1) or line.startswith(header2):
                line = lines.next()
                while  line.strip() != ('-10'):
                    result = line.split()
                    # check if its busted
                    if 'nan' in result or '-nan' in result:
                        self.create_debug()
                        self.create_mesh("broke")
                        log_error("structural failure", self.program)
                        print "***********STRUCTURE FAILED*************"
                        xx = int(DEFAULT_FIT)
                        xy = int(DEFAULT_FIT)
                        zx = int(DEFAULT_FIT)
                        stress = {'xx': xx, 'xy': xy, 'zx': zx}
                        stress_list.append(stress)
                        exit()
                        break
                    else:
                        xx = int(float(result[2]))
                        xy = int(float(result[3]))
                        zx = int(float(result[4]))
                        stressid = int(result[0])
                        stress = {'id': stressid, 'xx': xx, 'xy': xy, 'zx': zx}
                        stress_list.append(stress)
                        line = lines.next()
        results.close()
        return stress_list

    def run_optimization(self):
        optimizer = OPT.Optimizer(self.unique_id,self.program)
        optimizer.optimize_size(self.unique_id,button=False)

    def calculate_fitness(self):
        """return values for length and average stress on the structure"""
        total_length = (0)
        total_weight = (0)
        max_displacement = 0
        for node in self.node_list:
            for point in self.new_node_list:
                if node['id'] == point['id']:
                    displacement = three_d_line_length(node,point)
                    if displacement > max_displacement:
                        max_displacement = displacement
        for edge in self.edge_list:
            weight = float(edge['mass'])/10 # weight is in kilograms (mass in newtons)
            total_length = float(total_length) + float(edge['length'])  
            total_weight = float(total_weight) + float(weight)
        beams = len(self.edge_list)   
        compliance = float(self.compliance) 
        if hasattr(self.my_graph, 'valid') and self.my_graph.valid is False:
            print(__name__, "calculate_fitness", hasattr(self.my_graph, 'valid'), self.my_graph.valid, max_displacement, compliance, total_weight, DEFAULT_FIT)
            max_displacement = DEFAULT_FIT
            compliance = DEFAULT_FIT
            total_weight = DEFAULT_FIT
        if float(max_displacement) > self.max_displacement:
            max_displacement = DEFAULT_FIT
            compliance = DEFAULT_FIT
            total_weight = DEFAULT_FIT
        return compliance, max_displacement, total_weight

    def normalize(self, value, new_range, old_range):
        if value > old_range[1]:
            value = old_range[1]
        normalized = ((new_range[0] + (value - old_range[0])
                       * (new_range[1] - new_range[0]))
                      / old_range[1] - old_range[0])
        return normalized

    def test_mesh(self):
        """ calls all the functions in analyser"""
        self.create_graph()
        self.assign_load_case()
    #    self.assign_wind_loads()
        self.apply_stresses()
        self.create_slf_file()
        self.test_slf_file()
        if SHOW_ANALYSIS:
            self.show_analysis()
        if OPTIMIZE:
            self.run_optimization()
        self.parse_results()
        return self.calculate_fitness()       

    def show_mesh(self):
        """generate and show mesh stresses using bmpost"""
        self.create_graph()
        self.assign_load_case()
    #    self.assign_wind_loads()
        self.apply_stresses()
        self.create_slf_file()
        self.test_slf_file()
        self.parse_results()
        self.show_analysis()

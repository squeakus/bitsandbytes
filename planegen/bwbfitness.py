"""
generate a craft and then apply openfoam CFD to calculate lift and drag
"""
import elementtree.ElementTree as ET
import random, subprocess, os, time, copy
class CFD_Fitness():
    """Fitness function calculates lift and drag of a model.
    Parse the baseplane xml file, change some of the settings
    and then output a new xml file"""

    def __call__(self, program):
        exec(python_filter(program))
        starttime = time.time()
        self.scale_inputs()
        self.generate_xml()
        self.generate_stl()
        if self.debug:
            self.plane['lift'] = random.randint(0,100)
            self.plane['drag'] = random.randint(0,100)
        else:
            self.run_foam()
            forces = get_forces()
            self.plane['lift'] = forces['lift']
            self.plane['drag'] = forces['drag']

            
        self.plane['time'] = int(time.time() - starttime)
        return copy.deepcopy(self.plane)
    
    def __init__(self, debug=False, foampng=False):
        self.foampng = foampng
        self.debug = debug
        self.plane_cnt = 0
        self.maximise = True # false = smaller is better
        self.tree = ET.parse('planes/bwb.vsp')
        self.root = self.tree.getroot()
        self.plane = {}
        self.ranges = {}
        self.xml = {}

        #self.parts = ['section0', 'section1']
        self.parts = ['foil0', 'foil1', 'foil2']
        #self.parts = ['section0', 'section1', 'foil0', 'foil1', 'foil2']

        self.ranges['section0'] = {'Span':(3.0, 5.0),
                                   'TC':(4.0, 15.0), 
                                   'RC':(11.0, 25.0),
                                   'Sweep':(45.0, 60.0),
                                   'Dihedral':(0.0,30.0)}
        
        self.ranges['section1'] = {'Span':(3.0, 10.0),
                                   'TC':(0.0, 6.0),
                                   'RC':(7.0, 14.0),
                                   'Sweep':(20.0, 45.0),
                                   'Dihedral':(-10.0,20)}
        
        self.ranges['foil0'] = {'Camber':(0, 0.5), 'Camber_Loc':(0, 0.9),
                                'Thickness':(0.0, 0.5)}
        self.ranges['foil1'] = {'Camber':(0, 0.5), 'Camber_Loc':(0, 0.9),
                                'Thickness':(0.0, 0.5)}
        self.ranges['foil2'] = {'Camber':(0, 0.5), 'Camber_Loc':(0, 0.9),
                                'Thickness':(0.0, 0.5)}

        sect_cnt, foil_cnt = 0, 0
        for elem in self.root.getiterator():
            
            if elem.tag == "Section":
                name = 'section' + str(sect_cnt)
                self.xml[name] = elem
                sect_cnt += 1
                
            if elem.tag == "Airfoil":
                name = 'foil' + str(foil_cnt)
                self.xml[name] = elem
                foil_cnt += 1

    def scale_inputs(self):
        for partname in self.parts:
            part = self.plane[partname]

            for key in part:
                #if not key == "Dihedral": 
                minval = self.ranges[partname][key][0] 
                maxval = self.ranges[partname][key][1]
                part[key] = (part[key] * (maxval - minval)) + minval

    def generate_xml(self):
        for partname in self.parts:
            part = self.plane[partname]
            
            for key in part:
                elem = self.xml[partname]
                
                for child in elem.getchildren():
                    if child.tag == key:
                        child.text = str(part[key])

    def generate_stl(self, idx=None):
        self.tree.write("planes/plane.vsp")    
        run_cmd("vsp -batch planes/plane.vsp -stereo")
        offset_zaxis(15.0)
        idx = self.plane_cnt
        self.plane_cnt += 1

        if self.debug:
            if idx == None: planeid = "planes/plane.png"
            else: planeid = "planes/plane%03d"%idx
            self.tree.write(planeid+".vsp")
            run_cmd("vsp -batch "+planeid+".vsp -stereo")
            offset_zaxis(15.0, planeid)
            run_cmd("openscad scripts/genpng.scad --camera=-25,5,1,45.20,0,120,180 --imgsize=500,300 -o "+planeid+".png")

    def run_foam(self):
        """run the openfoam cfd suite"""
        os.chdir('cfd')
        run_cmd('./Allclean')
        run_cmd('./Allrun')
       
        if self.foampng: 
            name = "../planes/foam%03d"%self.plane_cnt 
            run_cmd('pvbatch --use-offscreen-rendering batchimage.py '+name)
        os.chdir('../')
            
###############Utility functions#################

def get_forces():
    cfdfile = 'cfd/postProcessing/forceCoeffs1/0/forceCoeffs.dat'
    if os.path.exists(cfdfile):
        """Parse the lift and drag from the postprocessing file"""
        forcefile = open(cfdfile)
        for line in forcefile:
            if not line.startswith('#'):
                line = line.strip()
                info = line.split('\t')
                info = [float(strval) for strval in info]
                forces = {'time':info[0], 
                          'pitchmoment':info[1],
                          'drag':info[2],
                          'lift':info[3], 
                          'frontlift':info[4], 
                          'rearlift':info[5]}
    else:
        print "Failed to write coefficient file"
        forces = {'time':0, 
                  'pitchmoment':0.0,
                  'drag':5.0,
                  'lift':0.0, 
                  'frontlift':0.0, 
                  'rearlift':0.0}        
    return forces

def run_cmd(cmd, debug = False):
    """execute commandline command cleanly"""
    if debug:
        print cmd
    else:
        cmd += " > /dev/null 2>&1"
    process = subprocess.Popen(cmd, shell=True,
                               stdout=subprocess.PIPE,
                               stdin=subprocess.PIPE)
    result = process.communicate()
    return result
    
def offset_zaxis(offset, name="planes/plane"):
    """Plane needs to be clear of the bottom of the wind tunnel
    there must be an easier way than this hack"""
    planename = name + '.stl'
    vertsymbol = 'vertex'
    stlfile = open(planename,'r')
    newstl = []
    
    for line in stlfile:
        tmpline = line.strip()
        if tmpline.startswith(vertsymbol):
            info = tmpline.split(' ')
            z = float(info[3])
            z += offset
            newline = info[0] + " " + info[1] + " " + info[2]
            newline += " " + str(z) + "\n"
            newstl.append(newline)
        else:
            newstl.append(line)

    outfile = open(planename, 'w')
    for line in newstl:
        outfile.write(line)
    outfile.close()

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
    """Create correct python syntax. We use {: and :} as special open
    and close brackets, because it's not possible to specify
    indentation correctly in a BNF grammar without this type of
    scheme."""

    indent_level = 0
    tmp = txt[:]
    i = 0
    while i < len(tmp):
        tok = tmp[i:i+2]
        if tok == "{:":
            indent_level += 1
        elif tok == ":}":
            indent_level -= 1
        tabstr = "\n" + "  " * indent_level
        if tok == "{:" or tok == ":}":
            tmp = tmp.replace(tok, tabstr, 1)
        i += 1
    # Strip superfluous blank lines.
    txt = "\n".join([line for line in tmp.split("\n")
                     if line.strip() != ""])
    return txt

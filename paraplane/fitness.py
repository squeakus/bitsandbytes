"""
generate a craft and then apply openfoam CFD to calculate lift and drag
"""
import elementtree.ElementTree as ET
import random, subprocess, os, copy

class Fitness():
    """Fitness function calculates lift and drag of a model.
    Parse the baseplane xml file, change some of the settings
    and then output a new xml file"""

    def __call__(self, program, idx):
        exec(python_filter(program))
        self.scale_inputs()
        self.generate_xml()
        self.generate_stl(idx)
        self.generate_cfd(idx)
        self.run_foam(idx)
        return copy.deepcopy(self.plane)

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

        if idx == None: planeid = "planes/plane.png"
        else: planeid = "planes/plane%03d"%idx
                
        self.tree.write(planeid+".vsp")
        run_cmd("vsp -batch "+planeid+".vsp -stereo")
        offset_zaxis(15.0, planeid)
        run_cmd("rm -rf "+planeid+".vsp")

    def generate_cfd(self, idx):
        planeid = "planes/plane%03d.stl" % idx
        cfdid = "cfd%03d" % idx
        run_cmd("rm -rf "+cfdid)
        run_cmd("cp -r cfd "+cfdid)
        run_cmd("cp "+planeid+" "+cfdid+"/constant/triSurface/plane.stl")
        
    def run_foam(self, idx):
        """run the openfoam cfd suite"""
        cfdid = "cfd%03d" % idx
        os.chdir(cfdid)
        #run_cmd('./Allclean') not needed for parallel execution
        #runname = "planegen"+str(self.runid)
        #run_cmd('qsub -N '+runname+' foam.pbs')
       
        if self.foampng: 
            name = "../planes/foam%03d"%self.plane_cnt 
            run_cmd('pvbatch --use-offscreen-rendering batchimage.py '+name)
        os.chdir('../')
            
###############Utility functions#################


def run_cmd(cmd, debug=False, qstat=False):
    """execute commandline command cleanly"""
    if debug:
        print cmd
    elif not qstat:
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

"""
generate a craft and then apply openfoam CFD to calculate lift and drag
"""
import elementtree.ElementTree as ET
import random, subprocess, os, time


def main():
    #random.seed(1)
    for i in range(1):
    	starttime = time.time()
    	values = generate_craft()
        print "generating plane", i
    	generate_stl(i)
    	#run_foam()
    	#forces = get_forces()
    	timetaken = int(time.time() - starttime)
    	#write_result(forces, values, timetaken)
        #print i,"lift:", str(forces['lift']), "drag:",str(forces['drag']),"time:", timetaken, "secs"

def generate_craft():
    """Parse the baseplane xml file, change some of the settings
    and then output a new xml file"""
    tree = ET.parse('planes/bwb.vsp')
    root = tree.getroot()
    sections = []
    airfoils = []
    section_vars = ['Span', 'TC', 'RC', 'Sweep','Dihedral']
    foil_vars = ['Camber', 'Thickness']
    
    for elem in root.getiterator():
        if elem.tag == "Section":
           sections.append(elem)
        if elem.tag == "Airfoil":
            airfoils.append(elem)
    sect0_ranges = [(3.0, 5.0), (4.0, 15.0), (11.0, 25.0), (45.0, 60.0),(0.0,30.0)]
    sect1_ranges = [(3.0, 10.0), (0.0, 6.0), (7.0, 14.0), (20.0, 45.0),(-10.0,20)]
    foil0_ranges = [(0, 0.05), (0.1, 0.2)]
    foil1_ranges = [(0, 0.1), (0.05, 0.2)]
    foil2_ranges = [(0, 0.1), (0.05, 0.4)]

    sect0_vals = mutate(sections[0], section_vars, sect0_ranges, tree)
    sect1_vals = mutate(sections[1], section_vars, sect1_ranges, tree)
    foil0_vals = mutate(airfoils[0], foil_vars, foil0_ranges, tree)
    foil1_vals = mutate(airfoils[1], foil_vars, foil1_ranges, tree)
    foil2_vals = mutate(airfoils[2], foil_vars, foil2_ranges, tree)
    
    tree.write("planes/plane.vsp")
    return [sect0_vals, sect1_vals, foil0_vals, foil1_vals, foil2_vals]
    
def mutate(section, variables, section_ranges, tree):
    """change wing section values in the parse tree"""
    new_vals = {}
    for idx, var in enumerate(variables):
        for elem in section.getchildren():
            if elem.tag == var:
                minval, maxval = section_ranges[idx]
                newval = random.uniform(minval, maxval)
                new_vals[var] = newval
                elem.text = str(newval)
    return new_vals


def generate_stl(idx=None):
    """convert the vsp-xml file to an stl using vsp, also save an image"""    
    run_cmd("vsp -script scripts/exportstl.vscript")
    offset_zaxis(15.0)

    if idx == None:
        planename = "planes/plane.png"
    else:
        planename = "planes/plane"+str(idx)+".png"

    run_cmd("openscad scripts/genpng.scad --imgsize=500,500 -o "+planename)


def run_foam():
    """run the openfoam cfd suite"""
    os.chdir('cfd')
    run_cmd('./Allclean')
    run_cmd('./Allrun')
    os.chdir('../')

def get_forces():
    """Parse the lift and drag from the postprocessing file"""
    forcefile = open('cfd/postProcessing/forceCoeffs1/0/forceCoeffs.dat')
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
    return forces

def write_result(forces, values, timetaken):
    """save info for recreating individual"""
    result = "{'lift':" + str(forces['lift']) 
    result += ", 'drag':" + str(forces['drag']) 
    result += ", 'sect0':" + str(values[0]) 
    result += ", 'sect1':" + str(values[1])
    result += ", 'foil0':" + str(values[2]) 
    result += ", 'foil1':" + str(values[3]) 
    result += ", 'foil2':" + str(values[4])
    result += ", 'time':" + str(timetaken) + "}\n"
    resultfile = open('results.txt','a')
    resultfile.write(result)
    resultfile.close()

###############Utility functions#################
    
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
    
def offset_zaxis(offset):
    """Plane needs to be clear of the bottom of the wind tunnel
    there must be an easier way than this hack"""
    planename = 'planes/plane.stl'
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

if __name__=='__main__':
    main()

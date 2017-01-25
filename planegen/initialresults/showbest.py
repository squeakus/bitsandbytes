import elementtree.ElementTree as ET
#from planegenerator import generate_stl, get_forces, run_cmd
import os, sys

def recreate_craft(individual):
    """Parse the baseplane xml file, change some of the settings
    and then output a new xml file"""
    tree = ET.parse('planes/baseplane.vsp')
    root = tree.getroot()
    sections = []
    airfoils = []
    for foil in airfoils:
        print foil

    
    for elem in root.getiterator():
        if elem.tag == "Section":
           sections.append(elem)
        if elem.tag == "Airfoil":
            airfoils.append(elem)

    for key in individual['sect0']:
        for elem in sections[0].getchildren():
            if elem.tag == key:
                elem.text = str(individual['sect0'][key])

    for key in individual['sect1']:
        for elem in sections[1].getchildren():
            if elem.tag == key:
                elem.text = str(individual['sect1'][key])

    for key in individual['foil0']:
        for elem in airfoils[0].getchildren():
            if elem.tag == key:
                elem.text = str(individual['foil0'][key])

    for key in individual['foil1']:
        for elem in airfoils[1].getchildren():
            if elem.tag == key:
                elem.text = str(individual['foil1'][key])

    for key in individual['foil2']:
        for elem in airfoils[2].getchildren():
            if elem.tag == key:
                elem.text = str(individual['foil2'][key])

    tree.write("planes/plane.vsp")

def run_foam(name="best"):
    """run the openfoam cfd suite"""
    os.chdir('cfd')
    run_cmd('./Allclean')
    run_cmd('./Allrun')
    run_cmd('pvbatch --use-offscreen-rendering batchimage.py '+name)
    os.chdir('../')


def print_best(filename):
    print "best for", filename
    resultfile = open(filename, 'r')
    population = []
    bestlift = 0
    indivlift = None
    bestdrag = 100
    indivdrag = None
    timetotal = 0

    for line in resultfile:
        population.append(eval(line))

    for indiv in population:
        timetotal += indiv['time']
        if indiv['lift'] > bestlift:
            bestlift = indiv['lift']
            indivlift = indiv
        if indiv['drag'] < bestdrag:
            bestdrag = indiv['drag']
            indivdrag = indiv

    print "avr time", timetotal / len(population)
    print "best lift", bestlift
    print "best drag", bestdrag

print os.getcwd()
for filename in os.listdir(os.getcwd()):
    
    if filename.endswith('txt'):
        print_best(filename)
#recreate_craft(indivdrag)
#generate_stl(0)
#run_foam("bestdrag")
#forces = get_forces()
#print "lift:", str(forces['lift']), "drag:",str(forces['drag'])


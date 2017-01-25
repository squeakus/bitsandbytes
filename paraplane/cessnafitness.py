"""
generate a craft and then apply openfoam CFD to calculate lift and drag
"""
import elementtree.ElementTree as ET
import random, subprocess, os, copy 
from fitness import Fitness

class CFD_Fitness(Fitness):
    """Fitness function calculates lift and drag of a model.
    Parse the baseplane xml file, change some of the settings
    and then output a new xml file"""
    
    def __init__(self, runid=0, debug=False, foampng=False):
        self.runid = runid
        self.foampng = foampng
        self.debug = debug
        self.maximise = True # false = smaller is better
        self.tree = ET.parse('planes/cessna.vsp')
        self.root = self.tree.getroot()
        self.plane = {}
        self.ranges = {}
        self.xml = {}

        #self.parts = ['section0', 'section1']
        #self.parts = ['foil0', 'foil1', 'foil2']
        self.parts = ['section2', 'section3', 'foil2', 'foil3']

        self.ranges['section2'] = {'Span':(2.0, 6.0),
                                   'TC':(2.0, 7.0), 
                                   'RC':(2.0, 7.0),
                                   'Sweep':(-20.0, 17.0)}
        
        self.ranges['section3'] = {'Span':(2.0, 9.0),
                                   'TC':(2.0, 5.0),
                                   'RC':(2.0, 5.0),
                                   'Sweep':(-20.0, 20.0)}

        self.ranges['foil2'] = {'Camber':(0, 0.2), 'Thickness':(0.04, 0.2)}
        self.ranges['foil3'] = {'Camber':(0, 0.2), 'Thickness':(0.04, 0.2)}

        sect_cnt, foil_cnt = 0, 0
        wing = None
        for elem in self.root.getiterator():
            if elem.tag == 'Component':
                name = elem.find("General_Parms").find("Name")
                if name.text== "Wing":
                    wing = elem
                    break

        for elem in wing.getiterator():
            if elem.tag == "Section":
                name = 'section' + str(sect_cnt)
                self.xml[name] = elem
                sect_cnt += 1
                
            if elem.tag == "Airfoil":
                name = 'foil' + str(foil_cnt)
                self.xml[name] = elem
                foil_cnt += 1

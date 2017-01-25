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
        self.tree = ET.parse('planes/bwb.vsp')
        self.root = self.tree.getroot()
        self.plane = {}
        self.ranges = {}
        self.xml = {}

        #self.parts = ['section0', 'section1']
        #self.parts = ['foil0', 'foil1', 'foil2']
        self.parts = ['section0', 'section1', 'foil0', 'foil1', 'foil2']

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
        
        self.ranges['foil0'] = {'Camber':(0, 0.05), 'Thickness':(0.1, 0.2)}
        self.ranges['foil1'] = {'Camber':(0, 0.1), 'Thickness':(0.05, 0.2)}
        self.ranges['foil2'] = {'Camber':(0, 0.1), 'Thickness':(0.05, 0.4)}

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

import sys
sys.path.append('/usr/lib/freecad/lib')

import FreeCAD, Part,  Sketcher, Mesh, random, subprocess
from FreeCAD import Base

def run_cmd(cmd, debug=False):
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

def make_car(imgname):
    docname = "randomcar"
    
    sketchnames = []
    App.newDocument(docname)
    App.setActiveDocument(docname)
    App.ActiveDocument=App.getDocument(docname)

    for i in range(10):
        offset = i * 5
        sname = "sketch"+str(i)
        sketchnames.append(sname)
        App.activeDocument().addObject('Sketcher::SketchObject',sname)
        App.activeDocument().getObject(sname).Placement = App.Placement(App.Vector(offset,0.000000,0.000000),
                                                                        App.Rotation(0.500000,0.500000,0.500000,0.500000))
        y1 = 15 + random.randint(-2,+2)
        App.ActiveDocument.getObject(sname).addGeometry(Part.Line(App.Vector(0.000000,0.000000,0),
                                                                  App.Vector(30.000000,y1,0)))
        App.ActiveDocument.getObject(sname).addConstraint(Sketcher.Constraint('Coincident',-1,1,0,1)) 

        y2 = 15 + random.randint(-2,+2)
        App.ActiveDocument.getObject(sname).addGeometry(Part.Line(App.Vector(30.000000,y1,0),
                                                                  App.Vector(50.000000,y2,0)))
        App.ActiveDocument.getObject(sname).addConstraint(Sketcher.Constraint('Coincident',0,2,1,1)) 


        #App.ActiveDocument.getObject(sname).addGeometry(Part.Line(App.Vector(50.000000,15.000000,0),App.Vector(71.760025,22.753183,0)))
        y3 = 22 + random.randint(-2,+2)
        App.ActiveDocument.getObject(sname).addGeometry(Part.Line(App.Vector(50.000000,y2,0),
                                                                  App.Vector(71.760025,y3,0)))
        App.ActiveDocument.getObject(sname).addConstraint(Sketcher.Constraint('Coincident',1,2,2,1)) 

        y4 = 25 + random.randint(-2,+2)
        App.ActiveDocument.getObject(sname).addGeometry(Part.Line(App.Vector(71.760025,y3,0),
                                                                  App.Vector(93.929802,y4,0)))
        App.ActiveDocument.getObject(sname).addConstraint(Sketcher.Constraint('Coincident',2,2,3,1)) 

        y5 = 29 + random.randint(-2,+2)
        App.ActiveDocument.getObject(sname).addGeometry(Part.Line(App.Vector(93.929802,y4,0),
                                                                  App.Vector(113.765884,y5,0)))
        App.ActiveDocument.getObject(sname).addConstraint(Sketcher.Constraint('Coincident',3,2,4,1)) 

        App.ActiveDocument.getObject(sname).addGeometry(Part.Line(App.Vector(113.765884,y5,0),
                                                                  App.Vector(113.474182,0.000000,0)))
        App.ActiveDocument.getObject(sname).addConstraint(Sketcher.Constraint('Coincident',4,2,5,1)) 

        App.ActiveDocument.getObject(sname).addGeometry(Part.Line(App.Vector(113.765884,0.000000,0),App.Vector(0.000000,-0.000000,0)))
        App.ActiveDocument.getObject(sname).addConstraint(Sketcher.Constraint('Coincident',5,2,6,1)) 
        App.ActiveDocument.getObject(sname).addConstraint(Sketcher.Constraint('Coincident',6,2,0,1)) 

        App.ActiveDocument.getObject(sname).addConstraint(Sketcher.Constraint('Horizontal',6))
        App.ActiveDocument.recompute()

    App.getDocument(docname).addObject('Part::Loft','Loft')
    App.getDocument(docname).getObject('Loft').Sections=[App.getDocument(docname).sketch9, App.getDocument(docname).sketch8, App.getDocument(docname).sketch7, App.getDocument(docname).sketch6, App.getDocument(docname).sketch5, App.getDocument(docname).sketch4, App.getDocument(docname).sketch3, App.getDocument(docname).sketch2, App.getDocument(docname).sketch1, App.getDocument(docname).sketch0, ]
    App.getDocument(docname).getObject('Loft').Solid=True
    App.getDocument(docname).getObject('Loft').Ruled=False
    App.ActiveDocument.recompute()

    __doc__=FreeCAD.getDocument(docname)
    __doc__.addObject("Part::Mirroring")
    __doc__.ActiveObject.Source=__doc__.getObject("Loft")
    __doc__.ActiveObject.Label="Loft (Mirror #1)"
    __doc__.ActiveObject.Normal=(1,0,0)
    __doc__.ActiveObject.Base=(0,0,0)
    del __doc__

    #App.ActiveDocument.recompute()
    #App.getDocument(docname).saveAs('./'+docname+'.fcstd')

    __objs__=[]
    __objs__.append(FreeCAD.getDocument(docname).getObject("Loft"))
    __objs__.append(FreeCAD.getDocument(docname).getObject("Part__Mirroring"))
    Mesh.export(__objs__,"./"+docname+".stl")
    del __objs__
    run_cmd("openscad genpng.scad --imgsize=500,500 -o "+imgname+".png", True)

    for sname in sketchnames:
        App.getDocument(docname).removeObject(sname)
    
    App.getDocument(docname).removeObject('Loft')    
    App.getDocument(docname).removeObject("Part__Mirroring")
    #run_cmd("gnome-open "+imgname+".png")

for i in range(20):
    imgname = "car"+str(i)
    make_car(imgname)

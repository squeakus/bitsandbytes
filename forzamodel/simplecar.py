import sys, random

# check if running through FreeCAD GUI or traditional Python
FREECADPATH='/usr/lib/freecad/lib'
freecad_gui = True
#if not(FREECADPATH in sys.path): # test based on PYTHONPATH
if not("FreeCAD" in dir()):       # test based on loaded module
  freecad_gui = False
print("freecad_gui:", freecad_gui)

if not(freecad_gui):
  print("add FREECADPATH to sys.path")
  sys.path.append(FREECADPATH)
  import FreeCAD

print("FreeCAD.Version:", FreeCAD.Version())
import Part
from FreeCAD import Base
import Sketcher
import Mesh

docname = "simplecar"
sketchnames = []
App.newDocument(docname)
App.setActiveDocument(docname)
App.ActiveDocument=App.getDocument(docname)

#import FreeCAD
#import Sketcher

for i in range(10):
    offset = i * 5
    sname = "sketch"+str(i)
    sketchnames.append(sname)
    App.activeDocument().addObject('Sketcher::SketchObject',sname)
    App.activeDocument().getObject(sname).Placement = App.Placement(App.Vector(offset,0.000000,0.000000),App.Rotation(0.500000,0.500000,0.500000,0.500000))
    App.ActiveDocument.getObject(sname).addGeometry(Part.Line(App.Vector(0.000000,0.000000,0),App.Vector(30.000000,15.000000,0)))
    App.ActiveDocument.getObject(sname).addConstraint(Sketcher.Constraint('Coincident',-1,1,0,1)) 
    App.ActiveDocument.recompute()
    App.ActiveDocument.recompute()
    App.ActiveDocument.getObject(sname).addGeometry(Part.Line(App.Vector(30.000000,15.000000,0),App.Vector(50.000000,15.000000,0)))
    App.ActiveDocument.getObject(sname).addConstraint(Sketcher.Constraint('Coincident',0,2,1,1)) 
    App.ActiveDocument.recompute()
    App.ActiveDocument.recompute()
    App.ActiveDocument.getObject(sname).addConstraint(Sketcher.Constraint('Horizontal',1)) 
    App.ActiveDocument.recompute()
    #App.ActiveDocument.getObject(sname).addGeometry(Part.Line(App.Vector(50.000000,15.000000,0),App.Vector(71.760025,22.753183,0)))
    y = (random.random()*10) + 12
    App.ActiveDocument.getObject(sname).addGeometry(Part.Line(App.Vector(50.000000,15.000000,0),App.Vector(71.760025,y,0)))
    
    App.ActiveDocument.getObject(sname).addConstraint(Sketcher.Constraint('Coincident',1,2,2,1)) 
    App.ActiveDocument.recompute()
    App.ActiveDocument.recompute()
    App.ActiveDocument.getObject(sname).addGeometry(Part.Line(App.Vector(71.760025,y,0),App.Vector(93.929802,26.837082,0)))
    App.ActiveDocument.getObject(sname).addConstraint(Sketcher.Constraint('Coincident',2,2,3,1)) 
    App.ActiveDocument.recompute()
    App.ActiveDocument.recompute()
    App.ActiveDocument.getObject(sname).addGeometry(Part.Line(App.Vector(93.929802,26.837082,0),App.Vector(113.765884,28.879032,0)))
    App.ActiveDocument.getObject(sname).addConstraint(Sketcher.Constraint('Coincident',3,2,4,1)) 
    App.ActiveDocument.recompute()
    App.ActiveDocument.recompute()
    App.ActiveDocument.getObject(sname).addGeometry(Part.Line(App.Vector(113.765884,28.879032,0),App.Vector(113.474182,0.000000,0)))
    App.ActiveDocument.getObject(sname).addConstraint(Sketcher.Constraint('Coincident',4,2,5,1)) 
    App.ActiveDocument.recompute()
    App.ActiveDocument.recompute()
    App.ActiveDocument.getObject(sname).addConstraint(Sketcher.Constraint('Vertical',5)) 
    App.ActiveDocument.recompute()
    App.ActiveDocument.getObject(sname).addGeometry(Part.Line(App.Vector(113.765884,0.000000,0),App.Vector(0.000000,-0.000000,0)))
    App.ActiveDocument.getObject(sname).addConstraint(Sketcher.Constraint('Coincident',5,2,6,1)) 
    App.ActiveDocument.getObject(sname).addConstraint(Sketcher.Constraint('Coincident',6,2,0,1)) 
    App.ActiveDocument.recompute()
    App.ActiveDocument.getObject(sname).addConstraint(Sketcher.Constraint('Horizontal',6))
    App.ActiveDocument.recompute()

App.getDocument('simplecar').addObject('Part::Loft','Loft')
App.getDocument('simplecar').getObject('Loft').Sections=[App.getDocument('simplecar').sketch9, App.getDocument('simplecar').sketch8, App.getDocument('simplecar').sketch7, App.getDocument('simplecar').sketch6, App.getDocument('simplecar').sketch5, App.getDocument('simplecar').sketch4, App.getDocument('simplecar').sketch3, App.getDocument('simplecar').sketch2, App.getDocument('simplecar').sketch1, App.getDocument('simplecar').sketch0, ]
App.getDocument('simplecar').getObject('Loft').Solid=True
#App.getDocument('simplecar').getObject('Loft').Ruled=True
App.getDocument('simplecar').getObject('Loft').Ruled=False
App.ActiveDocument.recompute()

__doc__=FreeCAD.getDocument(docname)
__doc__.addObject("Part::Mirroring")
__doc__.ActiveObject.Source=__doc__.getObject("Loft")
__doc__.ActiveObject.Label="Loft (Mirror #1)"
__doc__.ActiveObject.Normal=(1,0,0)
__doc__.ActiveObject.Base=(0,0,0)
del __doc__
App.ActiveDocument.recompute()

App.getDocument(docname).saveAs('/home/jonathan/freecadmodels/simplecar.fcstd')

__objs__=[]
__objs__.append(FreeCAD.getDocument("simplecar").getObject("Loft"))
__objs__.append(FreeCAD.getDocument("simplecar").getObject("Part__Mirroring"))
Mesh.export(__objs__,"/home/jonathan/freecadmodels/simplecar.stl")
del __objs__

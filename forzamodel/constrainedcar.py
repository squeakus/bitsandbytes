# Macro Begin: /home/jonathan/.FreeCAD/car4.FCMacro +++++++++++++++++++++++++++++++++++++++++++++++++
import sys

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

App.newDocument()
App.setActiveDocument("Unnamed")
App.ActiveDocument=App.getDocument("Unnamed")
#Gui.ActiveDocument=Gui.getDocument("Unnamed")

for i in range(10):
    offset = i * 10
    sname = "sketch"+str(i)
    
    App.activeDocument().addObject('Sketcher::SketchObject',sname)
    App.activeDocument().getObject(sname).Placement = App.Placement(App.Vector(offset, 0.00000,0.000000),App.Rotation(0.500000,0.500000,0.500000,0.500000))


    App.ActiveDocument.getObject(sname).addGeometry(Part.Line(App.Vector(0.000000,0.000000,0),App.Vector(35.000000,21.261019,0)))
    App.ActiveDocument.recompute()
    App.ActiveDocument.getObject(sname).addGeometry(Part.Line(App.Vector(35.000000,21.261019,0),App.Vector(77.602722,21.261019,0)))
    App.ActiveDocument.getObject(sname).addConstraint(Sketcher.Constraint('Coincident',0,2,1,1)) 
    App.ActiveDocument.recompute()
    App.ActiveDocument.recompute()
    App.ActiveDocument.getObject(sname).addConstraint(Sketcher.Constraint('Horizontal',1)) 
    App.ActiveDocument.recompute()
    App.ActiveDocument.getObject(sname).addGeometry(Part.Line(App.Vector(77.602722,21.261019,0),App.Vector(125.000000,43.585102,0)))
    App.ActiveDocument.getObject(sname).addConstraint(Sketcher.Constraint('Coincident',1,2,2,1)) 
    App.ActiveDocument.recompute()
    App.ActiveDocument.recompute()
    App.ActiveDocument.getObject(sname).addGeometry(Part.Line(App.Vector(125.000000,43.585102,0),App.Vector(165.000000,43.053577,0)))
    App.ActiveDocument.getObject(sname).addConstraint(Sketcher.Constraint('Coincident',2,2,3,1)) 
    App.ActiveDocument.recompute()
    App.ActiveDocument.recompute()
    App.ActiveDocument.getObject(sname).addConstraint(Sketcher.Constraint('Horizontal',3)) 
    App.ActiveDocument.recompute()
    App.ActiveDocument.getObject(sname).addGeometry(Part.Line(App.Vector(165.000000,43.585102,0),App.Vector(165.000000,-0.000000,0)))
    App.ActiveDocument.getObject(sname).addConstraint(Sketcher.Constraint('Coincident',3,2,4,1)) 
    App.ActiveDocument.recompute()
    App.ActiveDocument.recompute()
    App.ActiveDocument.getObject(sname).addConstraint(Sketcher.Constraint('PointOnObject',4,2,-1)) 
    App.ActiveDocument.recompute()
    App.ActiveDocument.getObject(sname).addConstraint(Sketcher.Constraint('Vertical',4)) 
    App.ActiveDocument.recompute()
    App.ActiveDocument.getObject(sname).addGeometry(Part.Line(App.Vector(165.000000,0.000000,0),App.Vector(1.063050,-0.000000,0)))
    App.ActiveDocument.getObject(sname).addConstraint(Sketcher.Constraint('Coincident',4,2,5,1)) 
    App.ActiveDocument.recompute()
    App.ActiveDocument.recompute()
    App.ActiveDocument.getObject(sname).addConstraint(Sketcher.Constraint('Coincident',5,2,-1,1)) 
    App.ActiveDocument.recompute()
    App.ActiveDocument.getObject(sname).addConstraint(Sketcher.Constraint('Horizontal',5)) 
    App.ActiveDocument.recompute()
    App.ActiveDocument.getObject(sname).addConstraint(Sketcher.Constraint('Distance',5,165.000000)) 
    App.ActiveDocument.getObject(sname).setDatum(11,165.000000)
    App.ActiveDocument.recompute()
    App.ActiveDocument.getObject(sname).addConstraint(Sketcher.Constraint('Distance',4,43.585102)) 
    App.ActiveDocument.getObject(sname).setDatum(12,43.585100)
    App.ActiveDocument.recompute()
    App.ActiveDocument.getObject(sname).addConstraint(Sketcher.Constraint('Distance',3,40.000000)) 
    App.ActiveDocument.getObject(sname).setDatum(13,40.000000)
    App.ActiveDocument.recompute()
    App.ActiveDocument.getObject(sname).addConstraint(Sketcher.Constraint('Distance',1,42.602722)) 
    App.ActiveDocument.getObject(sname).setDatum(14,42.602700)
    App.ActiveDocument.recompute()
    App.ActiveDocument.getObject(sname).addConstraint(Sketcher.Constraint('Angle',2,1,1,2,2.701414)) 
    App.ActiveDocument.getObject(sname).delConstraint(15)
    App.ActiveDocument.getObject(sname).movePoint(1,0,App.Vector(7.393383,0.000000,0),1)
    App.ActiveDocument.recompute()
    App.ActiveDocument.getObject(sname).delConstraint(10)
    App.ActiveDocument.getObject(sname).addConstraint(Sketcher.Constraint('Angle',2,1,1,2,2.632592)) 
    App.ActiveDocument.getObject(sname).setDatum(14,2.632585)
    App.ActiveDocument.recompute()
    App.ActiveDocument.getObject(sname).movePoint(0,0,App.Vector(2.064550,0.000000,0),1)
    App.ActiveDocument.recompute()
    App.ActiveDocument.getObject(sname).addConstraint(Sketcher.Constraint('Angle',0,2,1,1,2.655529)) 
    App.ActiveDocument.getObject(sname).setDatum(15,2.655536)
    App.ActiveDocument.recompute()
    App.ActiveDocument.getObject(sname).addConstraint(Sketcher.Constraint('Distance',0,47.391960)) 
    App.ActiveDocument.getObject(sname).setDatum(16,47.392000)
    App.ActiveDocument.recompute()
    App.ActiveDocument.getObject(sname).movePoint(1,0,App.Vector(0.000000,0.000000,0),1)
    App.ActiveDocument.recompute()
    App.ActiveDocument.getObject(sname).addConstraint(Sketcher.Constraint('Distance',2,44.008686)) 
    App.ActiveDocument.getObject(sname).setDatum(17,44.008700)
    App.ActiveDocument.recompute()
    print "FINISHED",i
App.getDocument('Unnamed').recompute()
App.getDocument("Unnamed").saveAs('/home/jonathan/freecadmodels/multicar.fcstd')

# Macro End: /home/jonathan/.FreeCAD/car4.FCMacro +++++++++++++++++++++++++++++++++++++++++++++++++

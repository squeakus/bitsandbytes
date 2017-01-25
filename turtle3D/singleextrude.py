# basic_extrusion_from_svg_file.py
# an example of an extrusion from a svg file
# created by charlyoleg on 2013/05/07
# license: CC BY SA 3.0

FREECADPATH='/usr/lib/freecad/lib' # adapt this path to your system

import sys
# choose your favorite test to check if you are running with FreeCAD GUI or traditional Python
freecad_gui = True
#if not(FREECADPATH in sys.path): # test based on PYTHONPATH
if not("FreeCAD" in dir()):       # test based on loaded module
  freecad_gui = False
print("dbg102: freecad_gui:", freecad_gui)

if not(freecad_gui):
  print("dbg101: add FREECADPATH to sys.path")
  sys.path.append(FREECADPATH)
  import FreeCAD

print("FreeCAD.Version:", FreeCAD.Version())
#FreeCAD.Console.PrintMessage("Hello from PrintMessage!\n") # avoid using this method because it is not printed in the FreeCAD GUI

import Part
from FreeCAD import Base

print("dbg111: start building the 3D part")

my_tmp_doc = FreeCAD.newDocument("doc_blabla") # you can create implicitly the document "doc_blabla" by using it!
import importSVG
importSVG.insert("Aphex_Twin_Logo.svg","doc_blabla")

## Two methods are possible:
# 1. Emulating the GUI control
# or
# 2. Using the Part module as soon as possible.

### 1. GUI control emulation
#my_tmp_doc.addObject("Part::Extrusion","Extrude")
##FreeCAD.getDocument("doc_blabla").addObject("Part::Extrusion","Extrude")  # alternative syntax
#my_tmp_doc.Extrude.Base = my_tmp_doc.path2992 # path2992 is the ID written in the svg file
##my_tmp_doc.Extrude.Base = FreeCAD.getDocument("doc_blabla").path2992 # alternative syntax
#my_tmp_doc.Extrude.Dir = (0,0,2)
#my_tmp_doc.Extrude.Solid = (True)
#my_tmp_doc.Extrude.TaperAngle = (0)
#my_tmp_doc.recompute() # create the shape
#
#my_solid = Part.makeSolid(my_tmp_doc.Extrude.Shape) # get a direct pointer to the shape
##my_solid = my_tmp_doc.Extrude.Shape # seems to be equivalent

## 2. Using the Part module earlier
my_svg_shape = my_tmp_doc.path2992.Shape
#my_svg_shape = FreeCAD.getDocument("doc_blabla").path2992.Shape # alternative syntax
my_svg_wire = Part.Wire(my_svg_shape.Edges)
my_svg_face = Part.Face(my_svg_wire)
# extrusion
my_solid = my_svg_face.extrude(Base.Vector(0,0,2)) # straight linear extrusion

## view and export your 3D part
output_stl_file="test_basic_extrusion_from_svg_file.stl"
Part.show(my_solid) # works only with FreeCAD GUI, ignore otherwise
my_solid.exportStl(output_stl_file)
print("output stl file: %s"%(output_stl_file))
#
print("dbg999: end of script")
#

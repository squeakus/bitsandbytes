# extrude_all_paths_from_svg_file.py
# a generic script that extrudes all paths of a svg file
# created by charlyoleg on 2013/05/08
# license: CC BY SA 3.0

FREECADPATH='/usr/lib/freecad/lib' # adapt this path to your system
input_svg_file="1j1t2014.svg" # 

import sys
# check if running through FreeCAD GUI or traditional Python
freecad_gui = True
#if not(FREECADPATH in sys.path): # test based on PYTHONPATH
if not("FreeCAD" in dir()):       # test based on loaded module
  freecad_gui = False
print("freecad_gui:", freecad_gui)

if not(freecad_gui):
  print("dbg101: add FREECADPATH to sys.path")
  sys.path.append(FREECADPATH)
  import FreeCAD

print("FreeCAD.Version:", FreeCAD.Version())
import Part
from FreeCAD import Base

print("dbg111: start building the 3D part")

my_tmp_doc = FreeCAD.newDocument("doc_0")
import importSVG
importSVG.insert(input_svg_file,"doc_0")

my_solids = []
for obj in my_tmp_doc.Objects:
  my_svg_shape = obj.Shape
  my_svg_wire = Part.Wire(my_svg_shape.Edges)
  my_svg_face = Part.Face(my_svg_wire)
  # straight linear extrusion
  my_solids.append(my_svg_face.extrude(Base.Vector(0,0,10))) 

my_compound = Part.makeCompound(my_solids)

## view and export your 3D part
output_stl_file="allpaths.stl"
Part.show(my_compound) # works only with FreeCAD GUI, ignore otherwise
my_compound.exportStl(output_stl_file)
print("output stl file: %s"%(output_stl_file))

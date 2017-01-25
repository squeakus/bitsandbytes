from vtkpython import *
from Tkinter import *
from vtkTkRenderWidget import *

root = Tk()

# create vtkTkRenderWidget
root.pane = vtkTkRenderWidget(root)

# pack the pane into the tk root
root.pane.pack(side = 'top', expand=1, fill = 'both',padx=3, pady=3)

ren = vtkRenderer()
renwin = root.pane.GetRenderWindow()
renwin.AddRenderer(ren)

p = vtkPoints()
p.SetNumberOfPoints(5)
p.SetPoint(0,[0.,0.,0.])
p.SetPoint(1,[2.,0.,0.])
p.SetPoint(2,[0.,2.,0.])
p.SetPoint(3,[0.,0.,2.])
p.SetPoint(4,[2.,2.,2.])

l = vtkCellArray()
l.Allocate(6,6)

l.InsertNextCell(2)
l.InsertCellPoint(0)
l.InsertCellPoint(1)

l.InsertNextCell(2)
l.InsertCellPoint(0)
l.InsertCellPoint(2)

l.InsertNextCell(2)
l.InsertCellPoint(0)
l.InsertCellPoint(3)

l.InsertNextCell(2)
l.InsertCellPoint(1)
l.InsertCellPoint(4)

l.InsertNextCell(2)
l.InsertCellPoint(2)
l.InsertCellPoint(4)

l.InsertNextCell(2)
l.InsertCellPoint(3)
l.InsertCellPoint(4)

col = vtkIntArray()
col.SetNumberOfComponents(1)
col.InsertNextTuple1(0)
col.InsertNextTuple1(1)
col.InsertNextTuple1(2)
col.InsertNextTuple1(3)
col.InsertNextTuple1(4)
col.SetName('col')

sizes = vtkFloatArray()
sizes.SetNumberOfComponents(1)
sizes.InsertNextTuple1(1.)
sizes.InsertNextTuple1(1.2)
sizes.InsertNextTuple1(1.4)
sizes.InsertNextTuple1(1.6)
sizes.InsertNextTuple1(1.8)
sizes.SetName('sizes')

poly = vtkPolyData()
poly.SetPoints(p)
poly.SetLines(l)
poly.GetPointData().AddArray(col)
poly.GetPointData().AddArray(sizes)

# If I use this, I can scale the glyphs
# poly.GetPointData().SetScalars(sizes)

# If I use this, I can colour the glyphs but not
# scale them using the sizes field
poly.GetPointData().SetScalars(col)

print poly

# look table (red/green/blue/gray/yellow)
t = vtkLookupTable()
t.SetNumberOfColors(5)
t.Build()
t.SetTableValue(0,1,0,0,1)
t.SetTableValue(1,0,1,0,1)
t.SetTableValue(2,0,0,1,1)
t.SetTableValue(3,.5,0.5,0.5,1)
t.SetTableValue(4,1,1,0,1)

m = vtkPolyDataMapper()
m.SetInput(poly)
m.SetScalarRange(0.0, 4.0)
m.SetLookupTable(t)
# this seems to work
m.SetScalarModeToUsePointFieldData()
m.ColorByArrayComponent('col',0)

a = vtkActor()
a.SetMapper(m)
ren.AddActor(a)

#
# Add sphere glyphs
#
s = vtkSphereSource()
g = vtkGlyph3D()
g.SetInput(poly)
g.SetSource(s.GetOutput())

# This works but the colour array is used for scaling
# as well
g.SetColorModeToColorByScalar()

# need to find a way of choosing a particular array
# there is no ScaleByArrayComponent
#g.SetScaleModeToScaleByScalar()
#?? g.ScaleByArrayComponent('col',0)
g.SetScaleModeToDataScalingOff()

m = vtkPolyDataMapper()
m.SetInput(g.GetOutput())
m.SetScalarRange(0.0, 4.0)
m.SetLookupTable(t)

# try and colour it
# this seems not to work either
#m.SetScalarModeToUsePointFieldData()
#m.ColorByArrayComponent('col',0)

print 'Glyph', g
print 'Glyph mapper',m
act = vtkActor()
act.SetMapper(m)
act.PickableOff()
ren.AddActor(act)

root.mainloop()

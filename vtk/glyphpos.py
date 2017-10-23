from vtk import *

# input data, every row is for a different item
positions = [[0, 0, 0],[1.5, 0, 0]]

orientations = [[1.0, 0.0, 0.0],[0.0, 1.0, 1.0]]

colors = [[255, 0, 0],
          [0, 255, 255]]

heights = [1,
           2]


# rendering of those two defined cylinders
points = vtkPoints()
points.InsertNextPoint(*positions[0])
points.InsertNextPoint(*positions[1])
polydata = vtkPolyData()
polydata.SetPoints(points)

color_def = vtkUnsignedCharArray()
color_def.SetNumberOfComponents(3)
color_def.SetNumberOfTuples(polydata.GetNumberOfPoints())
color_def.InsertTuple3(0, *colors[0])
color_def.InsertTuple3(1, *colors[1])
polydata.GetPointData().SetScalars(color_def)

pointNormalsArray = vtkDoubleArray()
pointNormalsArray.SetNumberOfComponents(3)
pointNormalsArray.SetNumberOfTuples(polydata.GetNumberOfPoints())
pointNormalsArray.SetTuple(0, orientations[0])
pointNormalsArray.SetTuple(1, orientations[1])
polydata.GetPointData().SetNormals(pointNormalsArray)

cyl_source = vtkCylinderSource()
cyl_source.SetResolution(10)
cyl_source.SetHeight(0.8)
cyl_source.SetRadius(0.1)
cyl_source.Update()

glyph = vtkGlyph3D()
glyph.SetInputData(polydata)
glyph.SetSourceConnection(cyl_source.GetOutputPort())
glyph.SetColorModeToColorByScalar()
glyph.SetVectorModeToUseNormal()
glyph.ScalingOff()

mapper = vtkPolyDataMapper()
mapper.SetInputConnection(glyph.GetOutputPort())
actor = vtkActor()
actor.SetMapper(mapper)
ren = vtkRenderer()
ren.AddActor(actor)

renwin = vtk.vtkRenderWindow()
renwin.AddRenderer(ren)
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renwin)
renwin.Render()
iren.Initialize()
iren.Start()

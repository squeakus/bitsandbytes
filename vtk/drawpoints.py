#!/usr/bin/env python
 
import vtk
 
points = [[0.0, 0.0, 0.0],[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],
          [1.0, 1.0, 0.0],[0.0, 0.0, 1.0]]

 
pointarray = vtk.vtkPoints()
cellarray = vtk.vtkCellArray()

for point in points:
    pointId = pointarray.InsertNextPoint(point)
    cellarray.InsertNextCell(1)
    cellarray.InsertCellPoint(pointId)
  
# Create a polydata to store everything in
PolyData = vtk.vtkPolyData()
PolyData.SetPoints(pointarray)
PolyData.SetVerts(cellarray)

# Setup actor and mapper
mapper = vtk.vtkPolyDataMapper()
mapper.SetInput(PolyData) 
actor = vtk.vtkActor()
actor.SetMapper(mapper)
 
# Setup render window, renderer, and interactor
renderer = vtk.vtkRenderer()
renderWindow = vtk.vtkRenderWindow()
renderWindow.AddRenderer(renderer)
renderWindowInteractor = vtk.vtkRenderWindowInteractor()
renderWindowInteractor.SetRenderWindow(renderWindow)
renderer.AddActor(actor) 
renderWindow.Render()
renderWindowInteractor.Start()

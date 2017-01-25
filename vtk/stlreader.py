#!/usr/bin/env python

# This example demonstrates the use of vtkSTLReader to load data into
# VTK from a file.  This example also uses vtkLODActor which changes
# its graphical representation of the data to maintain interactive
# performance.

import vtk

# Create the reader and read a data file.  Connect the mapper and
# actor.
sr = vtk.vtkSTLReader()
sr.SetFileName("plane.stl")

stlMapper = vtk.vtkPolyDataMapper()
stlMapper.SetInputConnection(sr.GetOutputPort())

stlActor = vtk.vtkLODActor()
stlActor.SetMapper(stlMapper)

# Create the Renderer, RenderWindow, and RenderWindowInteractor
ren = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)

# Add the actors to the render; set the background and size
ren.AddActor(stlActor)
#ren.SetBackground(1.0, 1.0, 1.0)
renWin.SetSize(500, 500)

# Zoom in closer
ren.ResetCamera()
cam1 = ren.GetActiveCamera()
cam1.Zoom(1.4)

iren.Initialize()
renWin.Render()
iren.Start()

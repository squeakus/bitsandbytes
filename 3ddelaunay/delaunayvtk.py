#!/usr/bin/env python
import numpy as np
from scipy.spatial import Delaunay
import random, vtk

def main():
    
    #make points and delaunay
    ordered = []
    for i in range(5):
        for j in range(5):
            for k in range(2):
                print "before",i,j,k
                x = 1 + random.uniform(-0.002,+0.002)
                y = j + random.uniform(-0.002,+0.002)
                z = k + random.uniform(-0.002,+0.002)
                
                #x,y,z = i,j,k
                print "after",x,y,z
            ordered.append([x,y,z])
    points = np.array(ordered)
    tri = Delaunay(points)

    vtkpoints = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    nodecnt = 0
    # insert initial point
    vtkpoints.InsertNextPoint([0,0,0])
    print "pointlength:", len(points)
    print "trilength:", len(tri.points)

    for point in points:
        vtkpoints.InsertNextPoint(point)
        line = vtk.vtkLine()
        nodecnt += 1

    for tri in tri.simplices:
        a,b,c,d = tri
        #print "edges:", a,b,c,d
        line.GetPointIds().SetId(0,a)
        line.GetPointIds().SetId(1,b)
        lines.InsertNextCell(line)

        line.GetPointIds().SetId(0,b)
        line.GetPointIds().SetId(1,c)
        lines.InsertNextCell(line)


        line.GetPointIds().SetId(0,c)
        line.GetPointIds().SetId(1,d)
        lines.InsertNextCell(line)


        line.GetPointIds().SetId(0,d)
        line.GetPointIds().SetId(1,a)
        lines.InsertNextCell(line)


    # for node in tree.leaves: 
    #     edges = node.getedges()
       
    #     for edge in edges:
    #         x0,y0,z0 = edge[0]
    #         x1,y1,z1 = edge[1]

    #         vtkpoints.InsertNextPoint(edge[0])
    #         vtkpoints.InsertNextPoint(edge[1])

    #         line = vtk.vtkLine()
    #         line.GetPointIds().SetId(0,nodecnt)
    #         line.GetPointIds().SetId(1,nodecnt+1)
    #         lines.InsertNextCell(line)
    #         nodecnt += 2

    # Create a polydata to store everything in
    linesPolyData = vtk.vtkPolyData()

    # Add the vtkpoints to the dataset
    linesPolyData.SetPoints(vtkpoints)
  
        # Add the lines to the dataset
    linesPolyData.SetLines(lines)

    # create sphere source
    source = vtk.vtkSphereSource()
    source.SetCenter(0,0,0)
    source.SetRadius(2.0)

    
    # Setup actor and mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper2 = vtk.vtkPolyDataMapper()

    if vtk.VTK_MAJOR_VERSION <= 5:
        mapper.SetInput(linesPolyData)
        mapper2.SetInput(source.GetOutput())
    else:
        mapper.SetInputData(linesPolyData)
        mapper2.SetInputConnection(source.GetOutputPort())

    actor = vtk.vtkActor()
    actor2 = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor2.SetMapper(mapper2)
    
    # Setup render window, renderer, and interactor
    renderer = vtk.vtkRenderer()
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)
    renderer.AddActor(actor)
    #actor2.GetProperty().SetRepresentationToWireframe()
    #renderer.AddActor(actor2)
    renderWindow.Render()
    renderWindowInteractor.Start()

if __name__=='__main__':
    main()

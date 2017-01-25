#!/usr/bin/env python
 
import vtk
from octree import Octree, Node

def main():
    #circles = [(1.9,0,0,0),(1.0,0.95,0,0)]
    circles = [(2.0,0,0,0)]

    rootcube = [-4.0, -4.0, -4.0, 4.0, 4.0, 4.0]
    resolution = 0.1

    rootnode = Node(None, rootcube)
    tree = Octree(rootnode, resolution, circles)
    print len(tree.leaves)
    print('so many leaves:%d' % len(tree.leaves))

    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    nodecnt = 0

    for node in tree.leaves: 
        edges = node.getedges()
       
        for edge in edges:
            x0,y0,z0 = edge[0]
            x1,y1,z1 = edge[1]

            points.InsertNextPoint(edge[0])
            points.InsertNextPoint(edge[1])

            line = vtk.vtkLine()
            line.GetPointIds().SetId(0,nodecnt)
            line.GetPointIds().SetId(1,nodecnt+1)
            lines.InsertNextCell(line)
            nodecnt += 2

    # Create a polydata to store everything in
    linesPolyData = vtk.vtkPolyData()

    # Add the points to the dataset
    linesPolyData.SetPoints(points)
  
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

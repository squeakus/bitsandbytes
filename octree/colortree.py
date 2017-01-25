#!/usr/bin/env python
 
import vtk
from octree import Octree, Node

renderer = vtk.vtkRenderer()
renderWindow = vtk.vtkRenderWindow()
    
def draw_lines(nodes, color):
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    nodecnt = 0
    colors = vtk.vtkUnsignedCharArray()
    colors.SetNumberOfComponents(3)
    colors.SetName("Colors")
    
    for node in nodes: 
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
            colors.InsertNextTupleValue(color)
            
    # Create a polydata to store everything in
    linesPolyData = vtk.vtkPolyData()
    # Add the points to the dataset
    linesPolyData.SetPoints(points)
    # Add the lines to the dataset
    linesPolyData.SetLines(lines)
    linesPolyData.GetCellData().SetScalars(colors)
    return linesPolyData

def draw_sphere(r,x,y,z):
    # create sphere source
    source = vtk.vtkSphereSource()
    source.SetCenter(x,y,z)
    source.SetRadius(r)
    return source

def branches():
    actors = renderer.GetActors()
    actors.InitTraversal()
    actor = actors.GetNextItem()
    while actor:
        if actor.name == "branches":
            if actor.shown == True:
                actor.VisibilityOff()
                actor.shown = False
            else:
                actor.VisibilityOn()
                actor.shown = True
        actor = actors.GetNextItem()
    renderWindow.Render()   
        
def keypress(obj, event):
    key = obj.GetKeySym()
    if key == "e":
        obj.InvokeEvent("DeleteAllObjects")
        sys.exit()
    elif key == "b":
        branches()
    
def main():
    #circles = [(1.9,0,0,0),(1.0,0.95,0,0)]
    circles = [(2.0,0,0,0)]
    rootcube = [-4.0, -4.0, -4.0, 4.0, 4.0, 4.0]
    resolution = 0.1
    green = [0, 255, 0]
    red = [255, 0, 0]
    blue = [0, 0, 255]
    
    rootnode = Node(None, rootcube)
    tree = Octree(rootnode, resolution, circles)
    print('so many leaves:%d' % len(tree.branches))
    objects = {}
    
    objects['branches'] = draw_lines(tree.branches,green)
    objects['leaves'] = draw_lines(tree.leaves,red)

    #objects.append(draw_sphere(2.0,0,0,0))
    
    # Setup render window, renderer, and interactor
    #renderer = vtk.vtkRenderer()
    renderWindow.AddRenderer(renderer)
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)
    renderWindowInteractor.AddObserver("KeyPressEvent", keypress)

    for name in objects:
        obj = objects[name]
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInput(obj)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.name = name
        actor.shown = True
        renderer.AddActor(actor)

    renderWindow.Render()
    renderWindowInteractor.Start()

if __name__=='__main__':
    main()

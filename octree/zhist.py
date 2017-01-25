#!/usr/bin/env python
# 26 mins for 0.1 res
# 0.2 res
#Elapsed time   : 7m47.300s
#User mode      : 7m27.734s
#System mode    : 0m1.347s
#CPU percentage : 96.10


import vtk, pylab, numpy
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

def midpoint(pt_a, pt_b):
    x = (pt_b[0] + pt_a[0]) / 2
    y = (pt_b[1] + pt_a[1]) / 2
    z = (pt_b[2] + pt_a[2]) / 2
    mid = (x, y, z)
    return mid

def parse_ply(filename):
    plyfile = open(filename, 'r')
    points = []
    vertcount = 0
    line = ""

    while not line.startswith("end_header"):
        line = plyfile.readline().rstrip()
        if line.startswith("element vertex"):
            line = line.lstrip("element vertex")
            vertcount = int(line)
        
    for _ in range(vertcount):
        line = plyfile.readline()
        line = line.split(' ')
        x,y,z = float(line[0]),float(line[1]),float(line[2])
        
        if z > 5:
           points.append((x,y,z))
    
    print "no of verts:", vertcount
    return points

def get_maxmin(points):
    zpoints = []
    inf, ninf = float('inf'), float('-inf')
    ranges = {'maxx':ninf, 'minx':inf,
              'maxy':ninf, 'miny':inf,
              'maxz':ninf, 'minz':inf}
    
    for point in points:
        x, y, z = point
        zpoints.append(z)
        if x > ranges['maxx']: ranges['maxx'] = x
        if x < ranges['minx']: ranges['minx'] = x
        if y > ranges['maxy']: ranges['maxy'] = y
        if y < ranges['miny']: ranges['miny'] = y
        if z > ranges['maxz']: ranges['maxz'] = z
        if z < ranges['minz']: ranges['minz'] = z
    ranges['xrange'] = ranges['maxx'] - ranges['minx']
    ranges['yrange'] = ranges['maxy'] - ranges['miny']
    ranges['zrange'] = ranges['maxz'] - ranges['minz']

    zpoints.sort()
    bins = numpy.arange(-50, 100)
    freq, bins = numpy.histogram(zpoints, bins)
    for i in range(len(bins)-1):
        #if freq[i] > 1000:
        #    print "baselayer", bins[i]
        #    break
        print bins[i], freq[i]
    pylab.hist(zpoints,bins=100)
    pylab.show()
    return ranges    
    
def main():
    circles = [(2.0,0,0,0)]
    points = parse_ply('damesmall.ply')
    ranges = get_maxmin(points)
    
    rootcube = [ranges['minx'], ranges['miny'], ranges['minz'],
                ranges['maxx'], ranges['maxy'], ranges['maxz']+400]
        
    resolution = 20.0
    green = [0, 255, 0]
    red = [255, 0, 0]
    blue = [0, 0, 255]
    
    # rootnode = Node(None, rootcube)
    # tree = Octree(rootnode, resolution, circles, points)
    # print('so many leaves:%d' % len(tree.branches))
    # objects = {}
    
    #objects['branches'] = draw_lines(tree.branches,green)
    #objects['leaves'] = draw_lines(tree.leaves,red)
    
    # Setup render window, renderer, and interactor
    #renderer = vtk.vtkRenderer()
    #renderWindow.AddRenderer(renderer)
    #renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    #renderWindowInteractor.SetRenderWindow(renderWindow)
    #renderWindowInteractor.AddObserver("KeyPressEvent", keypress)

    # for name in objects:
    #     obj = objects[name]
    #     mapper = vtk.vtkPolyDataMapper()
    #     mapper.SetInput(obj)
    #     actor = vtk.vtkActor()
    #     actor.SetMapper(mapper)
    #     actor.name = name
    #     actor.shown = True
    #     renderer.AddActor(actor)

    # renderWindow.Render()
    # renderWindowInteractor.Start()

if __name__=='__main__':
    main()

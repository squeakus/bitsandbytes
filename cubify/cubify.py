import vtk
from octree import Octree, Node

#global vtkrenderer for visibility swap
renderer = vtk.vtkRenderer()
renderWindow = vtk.vtkRenderWindow()

def main():
    #points = parse_asc('building.asc')
    #points = parse_asc('richview.asc')
    points = parse_asc('church.asc')
    
    stats = get_stats(points)
    
    rootcube = [stats['minx'], stats['miny'], stats['minz'],
                stats['maxx'], stats['maxy'], stats['maxz']]
    
    rootnode = Node(None, rootcube)
    tree = Octree(rootnode, rootcube, points)
    objects = {}
    
    objects['branches'] = draw_lines(tree.branches)
    objects['leaves'] = draw_lines(tree.leaves)
    #objects['points'] = draw_points(points)
    #draw_cubes(tree.leaves)
    write_ply(tree.leaves)
    vtkrender(objects)

def vtkrender(objects):
    global renderer, renderWindow
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

def draw_cubes(nodes):
    for node in nodes:
        # create cube
        center = node.getcenter()
        lengths = node.getlengths()
        cube = vtk.vtkCubeSource()
        cube.SetCenter(center)
        cube.SetXLength(lengths[0])
        cube.SetYLength(lengths[1])        
        cube.SetZLength(lengths[2])

        # mapper
        cubeMapper = vtk.vtkPolyDataMapper()
        cubeMapper.SetInput(cube.GetOutput())
        
        # actor
        cubeActor = vtk.vtkActor()
        cubeActor.SetMapper(cubeMapper)

        # assign actor to the renderer
        renderer.AddActor(cubeActor)
    
def draw_lines(nodes):
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
            colors.InsertNextTupleValue(node.color)
            
    # Create a polydata to store everything in
    linesPolyData = vtk.vtkPolyData()
    # Add the points to the dataset
    linesPolyData.SetPoints(points)
    # Add the lines to the dataset
    linesPolyData.SetLines(lines)
    linesPolyData.GetCellData().SetScalars(colors)
    return linesPolyData

def draw_points(points):
    pointarray = vtk.vtkPoints()
    cellarray = vtk.vtkCellArray()

    for point in points:
        pointId = pointarray.InsertNextPoint(point)
        cellarray.InsertNextCell(1)
        cellarray.InsertCellPoint(pointId)

    # Create a polydata to store everything in
    pointsPolyData = vtk.vtkPolyData()
    pointsPolyData.SetPoints(pointarray)
    pointsPolyData.SetVerts(cellarray)
    return pointsPolyData
    
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

def write_ply(nodes):
    outfile = open("out.ply",'w')
    header = "ply\nformat ascii 1.0 \nelement vertex "+str(len(nodes) * 8)
    header += "\nproperty float x\nproperty float y\nproperty float z\n"
    header += "element face " + str(len(nodes)*6)
    header += "\nproperty list uchar uint vertex_indices\nend_header\n"
    #header += "end_header\n"
    outfile.write(header)

    for node in nodes:
        for vert in node.getvertices():
            x,y,z = vert
            outfile.write(str(x) + ' ' + str(y) + ' ' + str(z) + '\n')

    for idx, node in enumerate(nodes):
        offset = idx * 8
        f1 = "4 " + str(offset+3) +' '+str(offset+2) +' '+str(offset+1)+' '+str(offset)+'\n'
        outfile.write(f1)
        f2 = "4 " + str(offset+4) +' '+str(offset+5) +' '+str(offset+6)+' '+str(offset+7)+'\n'
        outfile.write(f2)
        f3 = "4 " + str(offset) +' '+str(offset+1) +' '+str(offset+5)+' '+str(offset+4)+'\n'
        outfile.write(f3)
        f4 = "4 " + str(offset+3) +' '+str(offset+2) +' '+str(offset+6)+' '+str(offset+7)+'\n'
        outfile.write(f4)
        f5 = "4 " + str(offset) +' '+str(offset+3) +' '+str(offset+7)+' '+str(offset+4)+'\n'
        outfile.write(f5)
        f6 = "4 " + str(offset+1) +' '+str(offset+2) +' '+str(offset+6)+' '+str(offset+5)+'\n'
        outfile.write(f6)
        
    outfile.close()
 
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
        points.append((x,y,z))
    plyfile.close()
    return points

def parse_asc(filename):
    ascfile = open(filename, 'r')
    points = []

    for line in ascfile:
        if not line.startswith('//'):
            line = line.split(' ')
            x, y, z = float(line[0]), float(line[1]), float(line[2])
            points.append((x,y,z))

    ascfile.close()
    return points

    
def get_stats(points):
    inf, ninf = float('inf'), float('-inf')
    stats = {'maxx':ninf, 'minx':inf,
              'maxy':ninf, 'miny':inf,
              'maxz':ninf, 'minz':inf}
    
    for point in points:
        x, y, z = point
        if x > stats['maxx']: stats['maxx'] = x
        if x < stats['minx']: stats['minx'] = x
        if y > stats['maxy']: stats['maxy'] = y
        if y < stats['miny']: stats['miny'] = y
        if z > stats['maxz']: stats['maxz'] = z
        if z < stats['minz']: stats['minz'] = z
    stats['xrange'] = stats['maxx'] - stats['minx']
    stats['yrange'] = stats['maxy'] - stats['miny']
    stats['zrange'] = stats['maxz'] - stats['minz']
    return stats    
    
if __name__=='__main__':
    main()

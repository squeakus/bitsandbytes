import vtk

def midpoint(pt_a, pt_b):
    x = (pt_b[0] + pt_a[0]) / 2
    y = (pt_b[1] + pt_a[1]) / 2
    z = (pt_b[2] + pt_a[2]) / 2
    mid = (x, y, z)
    return mid

def sierpinski(tri, depth):
    if depth > 1:
        triangles = []
        triangles.append((tri[0],midpoint(tri[0], tri[1]),
                          midpoint(tri[0], tri[2])))
        triangles.append((tri[1],midpoint(tri[1], tri[0]),
                          midpoint(tri[1], tri[2])))
        triangles.append((tri[2],midpoint(tri[2], tri[0]),
                          midpoint(tri[2], tri[1])))
        triangles.append((midpoint(tri[0], tri[1]),
                          midpoint(tri[1], tri[2]),
                          midpoint(tri[0], tri[2])))
        return triangles
    else:
        return [tri]


stlfile = open('turbine_wheel.stl','r')
points, triangles = [], []

for line in stlfile:
    if line.find('vertex') != -1:
        vert1 = line.split()
        vert2 = stlfile.next().split()
        vert3 = stlfile.next().split()
        x1,y1,z1 = float(vert1[1]),float(vert1[2]),float(vert1[3])
        x2,y2,z2 = float(vert2[1]),float(vert2[2]),float(vert2[3])
        x3,y3,z3 = float(vert3[1]),float(vert3[2]),float(vert3[3])
        triangle = [(x1,y1,z1),(x2,y2,z2),(x3,y3,z3)]
        subdiv = sierpinski(triangle, 3)
        for tri in subdiv:
            triangles.append(tri)
        triangles.append(triangle)
        
    for triangle in triangles:
        for point in triangle:
            points.append(point)

        
print "no of points:", len(points)

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

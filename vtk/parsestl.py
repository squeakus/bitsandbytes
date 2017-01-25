import vtk
stlfile = open('plane.stl','r')

points = []

for line in stlfile:
    if line.startswith('vertex'):
        line = line.split(' ')
        x,y,z = float(line[1]),float(line[2]),float(line[3])
        points.append([x,y,z])
        
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

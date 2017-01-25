
inputFile = 'test.txt'

import numpy, vtk

data = numpy.genfromtxt(inputFile, delimiter=' ')
assert(data.shape[1] == 6)

numberOfPoints = data.shape[0]

points = vtk.vtkPoints()
points.SetNumberOfPoints(numberOfPoints)

colors = vtk.vtkUnsignedCharArray()
colors.SetName("RGB255")
colors.SetNumberOfComponents(3)
colors.SetNumberOfTuples(numberOfPoints)

for i in xrange(numberOfPoints):
    points.SetPoint(i, data[i][:3])
    colors.SetTuple(i, data[i][3:])

polyData = vtk.vtkPolyData()
polyData.SetPoints(points)
polyData.GetPointData().AddArray(colors)

mask = vtk.vtkMaskPoints()
mask.SetOnRatio(1)
mask.GenerateVerticesOn()
mask.SingleVertexPerCellOn()
mask.SetInput(polyData)
mask.Update()

self.GetOutput().ShallowCopy(mask.GetOutput())

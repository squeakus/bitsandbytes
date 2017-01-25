"""Read and show point cloud."""
import vtk
from numpy import random, genfromtxt, size


class VtkPointCloud:
    """Visual toolkit object for rendering clouds."""

    def __init__(self, zMin=10.0, zMax=50.0, maxNumPoints=4e6):
        """Can only handle 1 million points."""
        self.maxNumPoints = maxNumPoints
        self.vtkPolyData = vtk.vtkPolyData()
        self.clearPoints()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInput(self.vtkPolyData)
        mapper.SetColorModeToDefault()
        mapper.SetScalarRange(zMin, zMax)
        mapper.SetScalarVisibility(1)
        self.vtkActor = vtk.vtkActor()
        self.vtkActor.SetMapper(mapper)

    def addPoint(self, point):
        if self.vtkPoints.GetNumberOfPoints() < self.maxNumPoints:
            pointId = self.vtkPoints.InsertNextPoint(point[:])
            self.vtkDepth.InsertNextValue(point[2])
            self.vtkCells.InsertNextCell(1)
            self.vtkCells.InsertCellPoint(pointId)
            self.Colors.InsertNextTuple3(255, 0, 0)
        else:
            r = random.randint(0, self.maxNumPoints)
            self.vtkPoints.SetPoint(r, point[:])
        self.vtkCells.Modified()
        self.vtkPoints.Modified()
        self.vtkDepth.Modified()

    def clearPoints(self):
        """Initialise mapper for adding points."""
        self.vtkPoints = vtk.vtkPoints()
        self.vtkCells = vtk.vtkCellArray()
        self.vtkDepth = vtk.vtkDoubleArray()
        self.vtkDepth.SetName('DepthArray')
        self.vtkPolyData.SetPoints(self.vtkPoints)
        self.vtkPolyData.SetVerts(self.vtkCells)
        self.vtkPolyData.GetPointData().SetScalars(self.vtkDepth)
        self.vtkPolyData.GetPointData().SetActiveScalars('DepthArray')
        self.Colors = vtk.vtkUnsignedCharArray()
        self.Colors.SetNumberOfComponents(3)
        self.Colors.SetName("Colors")


def load_data(filename, pointCloud):
    """Use numpy func for reading in the cloud."""
    data = genfromtxt(filename, dtype=float, skiprows=2, usecols=[0, 1, 2])

    for k in xrange(size(data, 0)):
        point = data[k]  # 20*(random.rand(3)-0.5)
        pointCloud.addPoint(point)
    #pointCloud.vtkPolyData.GetCellData().SetScalars(pointCloud.Colors)
    pointCloud.vtkPolyData.GetPointData().AddArray(pointCloud.Colors)
    pointCloud.vtkPolyData.Modified()
    pointCloud.vtkPolyData.Update()
    return pointCloud


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print 'Usage: xyzviewer.py itemfile'
        sys.exit()
    pointCloud = VtkPointCloud()
    pointCloud = load_data(sys.argv[1], pointCloud)


# Renderer
    renderer = vtk.vtkRenderer()
    renderer.AddActor(pointCloud.vtkActor)
# renderer.SetBackground(.2, .3, .4)
    renderer.SetBackground(0.0, 0.0, 0.0)
    renderer.ResetCamera()

# Render Window
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)

# Interactor
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)

# Begin Interaction
    renderWindow.Render()
    renderWindow.SetWindowName("XYZ Data Viewer:"+sys.argv[1])
    renderWindowInteractor.Start()

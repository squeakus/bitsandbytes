import vtk
reader = vtk.vtkXMLPolyDataReader()
path =  "/usr/share/VTKData/Data/cow.vtp" #Archive path
#path =  "TriangleSolidColor.vtp" #Archive path
reader.SetFileName(path)
reader.Update()
 
mapper = vtk.vtkPolyDataMapper()
mapper.SetInput(reader.GetOutput())
 
actor = vtk.vtkActor()
actor.SetMapper(mapper)

# create a rendering window and renderer
ren = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)

# create a renderwindowinteractor
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)

# assign actor to the renderer
ren.AddActor(actor)

# enable user interface interactor
iren.Initialize()
renWin.Render()
iren.Start()
                

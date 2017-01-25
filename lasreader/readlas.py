import liblas, vtk

def main():
    pointcloud = liblas.file.File('316500_234500.las',mode='r')
    print "no of points:", len(pointcloud)

    renderer = vtk.vtkRenderer()
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)
    renderWindowInteractor.AddObserver("KeyPressEvent", keypress)

    objects = []
    for point in pointcloud:
        objects.append(draw_sphere(0.05,point.x, point.y, point.z))

    print "finished parsing"
    for obj in objects:
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInput(obj.GetOutput())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.name = "Sphere"
        actor.shown = True
        renderer.AddActor(actor)

    renderWindow.Render()
    renderWindowInteractor.Start()

def draw_sphere(r,x,y,z):
    # create sphere source
    source = vtk.vtkSphereSource()
    source.SetCenter(x,y,z)
    source.SetRadius(r)
    return source


def keypress(obj, event):
    key = obj.GetKeySym()
    if key == "e":
        obj.InvokeEvent("DeleteAllObjects")
        sys.exit()
    
if __name__=='__main__':
    main()

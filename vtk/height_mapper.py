import vtk
import sys

def main(options):
    png_file = options[0]
    #Read png
    png_reader = read_png(png_file)
    #render image
    render_image(png_reader)

def read_png(png_file):
    png_reader = vtk.vtkPNMReader()
    png_reader.SetFileName(png_file)
    return png_reader

def render_image(png_reader):
    square = 8
    color_map = vtk.vtkLookupTable()
    color_map.SetNumberOfColors(16)
    color_map.SetHueRange(0, 0.667)

    magnitude = vtk.vtkImageMagnitude()
    magnitude.SetInputData(png_reader.GetOutput())

    geometry = vtk.vtkImageDataGeometryFilter()
    geometry.SetInputData(magnitude.GetOutput())

    warp = vtk.vtkWarpScalar()
    warp.SetInputData(geometry.GetOutput())
    warp.SetScaleFactor(0.25)

    merge = vtk.vtkMergeFilter()
    merge.SetGeometryInputData(warp.GetOutput())
    merge.SetScalarsData(png_reader.GetOutput())

    elevation_mtHood = vtk.vtkElevationFilter()
    elevation_mtHood.SetInputData(merge.GetOutput())
    elevation_mtHood.SetLowPoint(0, 0, 0)
    elevation_mtHood.SetHighPoint(0, 0, 50)

    mapper_3D_mtHood = vtk.vtkDataSetMapper()
    mapper_3D_mtHood.SetInputData(elevation_mtHood.GetOutput())
    mapper_3D_mtHood.SetLookupTable(color_map)

    mapper_2D_mtHood = vtk.vtkPolyDataMapper2D()
    mapper_2D_mtHood.SetInputData(elevation_mtHood.GetOutput())
    mapper_2D_mtHood.SetLookupTable(color_map)
    
    actor_2D_mtHood = vtk.vtkActor2D()
    actor_2D_mtHood.SetMapper(mapper_2D_mtHood)
    actor_2D_mtHood.GetPositionCoordinate().SetCoordinateSystemToNormalizedDisplay()
    actor_2D_mtHood.GetPositionCoordinate().SetValue(0.25,0.25)
    
    actor_3D_mtHood = vtk.vtkActor()
    actor_3D_mtHood.SetMapper(mapper_3D_mtHood)


    renderer = vtk.vtkRenderer()
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)

    renderer.AddActor(actor_3D_mtHood)
    renderer.SetBackground(.5, .5, .5)

    renderWindow.SetSize(600, 600)
    renderWindow.Render()
    renderWindowInteractor.Start()


if __name__ == "__main__":
    options = sys.argv[1:]
    main(options)

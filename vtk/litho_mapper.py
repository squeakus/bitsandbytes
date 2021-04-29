import pyvista as pv
import numpy as np
from pyvista import examples
import matplotlib.cm as cm
import cv2
import vtk
import numpy as np
from vtk.util import numpy_support


def main():
    # dem = examples.download_crater_topo()
    # dem = pv.read("OldMtStHelens.dem")

    # image = cv2.imread("wafer.png", 0)
    # print(image.shape)

    # dem = pv.UniformGrid()
    # dem.dimensions = 435, 456, 1
    # dem.cell_arrays["values"] = image.flatten(order="F") 
    # print(f"dem {type(dem)}")

    dem = pv.read("wafer2.png")
    #dem.plot(cpos="xy")
    terrain = dem.warp_by_scalar(factor = 0.2)
    #pv.save_meshio("mesh.obj", terrain)

    terrain.plot(cmap=cm.coolwarm, cpos="xy")
    z_cells = np.array([25] * 5 + [35] * 3 + [50] * 2 + [75, 100])

    xx = np.repeat(terrain.x, len(z_cells), axis=-1)
    yy = np.repeat(terrain.y, len(z_cells), axis=-1)
    zz = np.repeat(terrain.z, len(z_cells), axis=-1) - np.cumsum(z_cells).reshape((1, 1, -1))

    mesh = pv.StructuredGrid(xx, yy, zz)
    mesh["Elevation"] = zz.ravel(order="F")
    mesh.plot(cmap=cm.coolwarm, cpos="xy")  # show_edges=True


if __name__ == "__main__":
    main()
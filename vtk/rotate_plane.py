import pyvista as pv
from pyvista import examples

mesh = examples.load_airplane()

# Use `pv.Box()` or `pv.Cube()` to create a region of interest
roi = pv.Cube(center=(0.9e3, 0.2e3, mesh.center[2]), x_length=500, y_length=500, z_length=500)
roi.rotate_z(33)

p = pv.Plotter()
p.add_mesh(roi, opacity=0.75, color="red")
p.add_mesh(mesh, opacity=0.5)
p.show()
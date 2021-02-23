import pyvista as pv
from pyvista import examples

mesh = examples.download_knee_full()


p = pv.Plotter(shape=(1, 2))

# opacity = [0, 0, 0, 0.1, 0.3, 0.6, 1]
p.add_volume(mesh, cmap="bone", opacity="sigmoid")
# p.add_mesh_slice(
#     mesh,
# )
p.subplot(0, 1)

# p.show()

# p1 = pv.Plotter()
p.add_mesh_slice_orthogonal(mesh)
# p.link_views()
p.show()

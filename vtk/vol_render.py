import pyvista as pv
from pyvista import examples

# Download a volumetric dataset
vol = examples.download_knee_full()
cpos = [(-381.74, -46.02, 216.54), (74.8305, 89.2905, 100.0), (0.23, 0.072, 0.97)]
opacity = [0, 0, 0, 0.1, 0.3, 0.6, 1]
p = pv.Plotter(shape=(1,2))
p.add_volume(vol, cmap="viridis", opacity=opacity, shade=False)
p.add_text("No shading")
p.subplot(0,1)
p.add_volume(vol, cmap="viridis", opacity=opacity, shade=True)
p.add_text("Shading")
p.link_views()
p.camera_position = cposp.show()


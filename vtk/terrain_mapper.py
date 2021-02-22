import pyvista as pv
import numpy as np
from pyvista import examples


dem = examples.download_crater_topo()
print(type(dem))
subset = dem.extract_subset((500, 900, 400, 800, 0, 0), (5,5,1))
subset.plot(cpos="xy")
terrain = subset.warp_by_scalar()
terrain.plot()

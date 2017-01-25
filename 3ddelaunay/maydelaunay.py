from mayavi import mlab
import numpy as np
from scipy.spatial import Delaunay
import random

#create the points
ordered = []
for i in range(5):
    for j in range(5):
        for k in range(5):
            x = i + random.uniform(-0.2,+0.2)
            y = j + random.uniform(-0.2,+0.2)
            z = k + random.uniform(-0.2,+0.2)

    ordered.append([x,y,z])
points = np.array(ordered)


mlab.figure(1, bgcolor=(0, 0, 0))
mlab.clf()
x,y,z = points[:,0], points[:,1], points[:,2]
mlab.mesh(x,y,z)
#pts = mlab.points3d(x, y, z, scale_factor=0.015, resolution=10)
mlab.show()

#pts = mlab.points3d(points[:,0], points[:,1], points[:,1], 1.5 * scalars.max() - scalars, scale_factor=0.015, resolution=10)


#pts.mlab_source.dataset.lines = np.array(connections)

# Use a tube fiter to plot tubes on the link, varying the radius with the
# scalar value
#tube = mlab.pipeline.tube(pts, tube_radius=0.15)
#tube.filter.radius_factor = 1.
#tube.filter.vary_radius = 'vary_radius_by_scalar'
#mlab.pipeline.surface(tube, color=(0.8, 0.8, 0))

# Visualize the local atomic density
#mlab.pipeline.volume(mlab.pipeline.gaussian_splatter(pts))


#mlab.view(49, 31.5, 52.8, (4.2, 37.3, 20.6))

import numpy as np
from scipy.spatial import Delaunay
import random
#points = np.array([[0, 0], [0, 1.1], [1, 0], [1, 1]])
#points = np.random.rand(10, 2)
ordered = []
for i in range(5):
    for j in range(5):
        for k in range(5):
            x = i + random.uniform(-0.2,+0.2)
            y = j + random.uniform(-0.2,+0.2)
            z = k + random.uniform(-0.2,+0.2)

    ordered.append([x,y,z])
points = np.array(ordered)


tri = Delaunay(points)

import matplotlib.pyplot as plt
print points
plt.triplot(points[:,0], points[:,1], tri.simplices.copy())
plt.plot(points[:,0], points[:,1], 'o')


for j, p in enumerate(points):
    plt.text(p[0]-0.03, p[1]+0.03, j, ha='right') # label the points
for j, s in enumerate(tri.simplices):
    p = points[s].mean(axis=0)
    plt.text(p[0], p[1], '#%d' % j, ha='center') # label triangles
#plt.xlim(-0.5, 1.5); plt.ylim(-0.5, 1.5)
plt.show()


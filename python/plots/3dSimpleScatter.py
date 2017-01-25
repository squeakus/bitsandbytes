import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


fig = plt.figure()
ax = Axes3D(fig)


point_list = [[0,0,0],[0,0,10],[1,3,4],[10,10,10],[7,5,4]]

xs, ys, zs = [], [], []
for point in point_list:
    xs.append(point[0])
    ys.append(point[1])
    zs.append(point[2])

ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')


ax.scatter(xs, ys, zs, s=100 ,zdir='x', c='r', marker='+')


plt.show()

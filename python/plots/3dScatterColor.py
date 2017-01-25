import pylab as p
import mpl_toolkits.mplot3d.axes3d as p3

#data is an ndarray with the necessary data and colors is an ndarray with
#'b', 'g' and 'r' to paint each point according to its class

x=[1,2,3,4,5,6,7,8]
y=[1,2,3,4,5,6,7,8]
z=[1,2,3,4,5,6,7,8]

x1=[9,8,7,6,5,4,3,2,1]
y1=[1,2,3,4,5,6,7,8,9]
z1=[9,8,7,6,5,4,3,2,1]



fig=p.figure()
ax = p3.Axes3D(fig)
ax.scatter(x, y, z, c='b')
ax.scatter(x1, y1, z1, c='r')


ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
fig.add_axes(ax)
p.show()

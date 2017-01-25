""" read voxels and render them as cubes."""

import bpy
import numpy as np
from mathutils import Vector

#bpy.ops.object.camera_add()
bpy.ops.mesh.primitive_cube_add()

ob = bpy.context.object
obs = []
sce = bpy.context.scene
maxx, maxy, maxz, maxi = -np.inf, -np.inf, -np.inf, -np.inf
minx, miny, minz, mini = np.inf, np.inf, np.inf, np.inf
voxels = []
infile = open('/home/jonathan/data/Jonathan/programs/blenderscripts/smallvoxel.csv', 'r')

for line in infile:
    (x, y, z, i) = [int(x) for x in line.split(",")]
    voxels.append([x, y, z])
    if x > maxx:
        maxx = x
    if x < minx:
        minx = x
    if y > maxy:
        maxy = y
    if y < miny:
        miny = y
    if z > maxz:
        maxz = z
    if z < minz:
        minz = z
    if i > maxi:
        maxi = i
    if i < mini:
        mini = i

for voxel in voxels:
    copy = ob.copy()
    copy.location = Vector((voxel[0]-minx, voxel[1]-miny, voxel[2]-minz))
    copy.data = copy.data.copy()  # duplicate mesh, remove for linked duplicate
    obs.append(copy)

for ob in obs:
    sce.objects.link(ob)

sce.update()

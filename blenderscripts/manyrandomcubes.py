"""
Create more cubes faster.

Only create one and reference it.also dont use add_primitive because
it automatically calls update scene every time.
"""

import bpy
from random import randint
from mathutils import Vector

bpy.ops.mesh.primitive_cube_add()
# how many cubes you want to add
count = 10000

ob = bpy.context.object
obs = []
sce = bpy.context.scene

for i in range(0, count):
    x = randint(-500, 500)
    y = randint(-500, 500)
    z = randint(-500, 500)
    copy = ob.copy()
    copy.location = Vector((x, y, z))
    copy.data = copy.data.copy()  # duplicate mesh, remove for linked duplicate
    obs.append(copy)

for ob in obs:
    sce.objects.link(ob)

sce.update()

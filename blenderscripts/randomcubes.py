import bpy
from random import randint
bpy.ops.mesh.primitive_cube_add()

#how many cubes you want to add
count = 1000

for c in range(0,count):
    print "cube:",c
    x = randint(-500,500)
    y = randint(-500,500)
    z = randint(-500,500)
    bpy.ops.mesh.primitive_cube_add(location=(x,y,z))
"""Create a single block and move it."""
import bpy
import mathutils

bpy.ops.mesh.primitive_cube_add(radius=0.5)

for item in list(bpy.data.objects):
    if item.type == 'MESH':
        Cube = bpy.data.objects[item.name]
        Cube.show_name = True
        Cube.delta_location += mathutils.Vector((0, 0, 1))

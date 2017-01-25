"""
Create more cubes faster.

Only create one and reference it.also dont use add_primitive because
it automatically calls update scene every time.
"""

import bpy
from random import randint
from mathutils import Vector


def makeMaterial(name, diffuse, specular, alpha):
    """Material creator for blender render."""
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = diffuse
    mat.diffuse_shader = 'LAMBERT'
    mat.diffuse_intensity = 1.0
    mat.specular_color = specular
    mat.specular_shader = 'COOKTORR'
    mat.specular_intensity = 0.5
    mat.alpha = alpha
    mat.use_transparency = True
    mat.ambient = 1
    return mat


def setMaterial(ob, mat):
    """apply the material to the object."""
    me = ob.data
    me.materials.append(mat)


def main():
    """how many cubes you want to add."""
    bpy.ops.mesh.primitive_cube_add(radius=0.5)
    red = makeMaterial('Red', (1, 0, 0), (1, 1, 1), 0.1)
    blue = makeMaterial('Blue', (0, 0, 1), (0.5, 0.5, 0), 0.1)

    count = 100
    ob = bpy.context.object
    obs = []
    sce = bpy.context.scene

    for i in range(0, count):
        x = randint(-10, 10)
        y = randint(-10, 10)
        z = randint(-10, 10)
        copy = ob.copy()
        copy.location = Vector((x, y, z))
        copy.data = copy.data.copy()  # dup mesh, remove for linked duplicate
        if i % 2 == 0:
            setMaterial(copy, red)
        else:
            setMaterial(copy, blue)

        obs.append(copy)

    for ob in obs:
        sce.objects.link(ob)

    sce.update()


# One way is to assign tuples directly to the camera object's location and
# rotation_euler attributes. For example, with the camera selected:
# import bpy
# from math import radians
# camera = bpy.context.object
# camera.location = (1.0, 0.0, 1.0)
# camera.rotation_euler = (radians(45), 0.0, radians(30))


if __name__ == "__main__":
    main()

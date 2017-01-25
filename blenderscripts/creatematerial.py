"""how to create material."""
import bpy


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


def run(origin):
    """Create two materials."""
    red = makeMaterial('Red', (1, 0, 0), (1, 1, 1), 1)
    blue = makeMaterial('BlueSemi', (0, 0, 1), (0.5, 0.5, 0), 0.5)

    # Create red cube
    bpy.ops.mesh.primitive_cube_add(location=origin)
    setMaterial(bpy.context.object, red)
    # and blue sphere
    bpy.ops.mesh.primitive_uv_sphere_add(location=origin)
    bpy.ops.transform.translate(value=(1, 0, 0))
    setMaterial(bpy.context.object, blue)

if __name__ == "__main__":
    run((0, 0, 0))

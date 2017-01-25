bl_info = {
    "name": "Import Medit Meshes",
    "author": "Jonathan Byrne",
    "version": (0,1),
    "blender": (2, 5, 7),
    "api": 35853,
    #"location": "View3D > Tool Shelf > Align Tools Panel",
    'location': 'File > Import > Medit file (.mesh)',
    "description": "imports .mesh files",
    "warning": "",
    "wiki_url": "http://wiki.blender.org/index.php/Extensions:2.5/Py/"\
        "Scripts/3D interaction/Align_Tools",
    "tracker_url": "https://projects.blender.org/tracker/index.php?"\
        "func=detail&aid==22389",
    "category": "Import-Export"}
"""Import mesh"""

import os, math, geometry
from math import sin, cos, radians
import bpy
import mathutils
from mathutils import Vector, Matrix

# Method to draw shapes in the 3D window using Python code
def drawShapesEval(objs):
    if objs is None or len(objs) == 0:
        print("Problem: no objects to draw")

    else:
        # Clear the scene (make sure any leftover objects are gone before
        # drawing a new scene)
        scene = bpy.data.scenes.active
        obs = scene.objects
        for ob in obs:
            scene.objects.unlink(ob)        
        scene.update(0)
        
        # Redraw the scene
        for obj in objs:
            scene.objects.link(obj)      # Link the Objects to the scene

        join_all_meshes()
        Blender.Redraw()

def simple_connect(context, pt_a, pt_b):
    pt_a = [pt_a[0]/1000,pt_a[1]/1000,pt_a[2]/1000]
    pt_b = [pt_b[0]/1000,pt_b[1]/1000,pt_b[2]/1000]
    #attempt at creating mesh
    # if pt_a[1] < pt_b[1]:
    #     verts=[pt_a, pt_b, [pt_b[0], pt_b[1]+0.1, pt_b[2]+0.1], [pt_a[0], pt_a[1]+0.1, pt_a[2]+0.1]]
    # else:
    #     verts=[pt_a, pt_b, [pt_b[0], pt_b[1]-0.1, pt_b[2]+0.1], [pt_a[0], pt_a[1]-0.1, pt_a[2]+0.1]]
    verts=[pt_a, pt_b, pt_b, pt_a]
    faces= [[0,1,2,3]]
    my_mesh = bpy.data.meshes.new('myMesh')          # create a new mesh
    my_mesh.from_pydata(verts,[],faces)
    my_mesh.update()
    scene = context.scene
    obj_act = scene.objects.active
    ob_new = bpy.data.objects.new('myObj', my_mesh)
    # Link new object to the given scene and select it.
    scene.objects.link(ob_new)
    ob_new.select = True

def edge_connect(context, pt_a, pt_b):
    pt_a = [pt_a[0]/1000,pt_a[1]/1000,pt_a[2]/1000]
    pt_b = [pt_b[0]/1000,pt_b[1]/1000,pt_b[2]/1000]
    #attempt at creating mesh
    verts = [pt_a, pt_b]  
    faces = [[0,1]]
    my_mesh = bpy.data.meshes.new('myMesh')          # create a new mesh
    my_mesh.from_pydata(verts,faces, [])
    my_mesh.update()
    scene = context.scene
    obj_act = scene.objects.active
    ob_new = bpy.data.objects.new('myObj', my_mesh)
    # Link new object to the given scene and select it.
    scene.objects.link(ob_new)
    ob_new.select = True


class IMPORT_OT_medit_mesh(bpy.types.Operator):
    bl_idname = "import.medit_mesh"
    bl_label = "import medit mesh"
    bl_description = "Create blender object from mesh"
    bl_options = {'REGISTER', 'UNDO'}
    filepath = bpy.props.StringProperty(subtype="FILE_PATH")
    nodes = [[0,0,0]]
    edges = []

    def parse_mesh(self, context, filename):
        mesh_file = open(filename,'r')
        lines = iter(mesh_file)
        for line in lines:
            line = line.rstrip()
            if line == "Vertices":
                counter = int(lines.__next__())
                for i in range(counter):
                    line = lines.__next__().rstrip()
                    array = line.split(' ')
                    index = i + 1
                    xyz = (int(array[0]), int(array[1]), int(array[2]))
                    node = xyz
                    self.nodes.append(node)
            if line == 'Edges':
               counter = int(lines.__next__())
               for i in range(counter):
                   line = lines.__next__().rstrip()
                   array = line.split(' ')
                   edge = (int(array[0]), int(array[1]))
                   self.edges.append(edge)
        mesh_file.close()
        print("found", len(self.nodes), "nodes and", len(self.edges), "edges")
        beamlist = []
        for edge in self.edges:
          pt_a, pt_b = self.nodes[edge[0]], self.nodes[edge[1]]
          beamlist.append(simple_connect(context, pt_a, pt_b))

    def execute(self, context):
        print("filePath:", self.filepath)
        self.parse_mesh(context, self.filepath)
        #bpy.ops.mesh.primitive_cube_add()
        return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


def import_images_button(self, context):
    self.layout.operator_context = 'INVOKE_DEFAULT'
    self.layout.operator(IMPORT_OT_medit_mesh.bl_idname, text="Import Medit Mesh", icon='PLUGIN')

# registering and menu integration
def register():
    bpy.utils.register_class(IMPORT_OT_medit_mesh)
    #bpy.utils.register_module(__name__)
    bpy.types.INFO_MT_file_import.append(import_images_button)

# unregistering and removing menus
def unregister():
    bpy.utils.unregister_class(IMPORT_OT_medit_mesh)
    bpy.types.INFO_MT_file_import.remove(import_images_button)
 
if __name__ == "__main__":
    register()

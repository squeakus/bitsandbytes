bl_info = {
    'name': 'Import MEDIT files from medit',
    'author': 'Jonathan Byrne',
    'version': (0, 1, 5),
    "blender": (2, 5, 7),
    "api": 36079,
    'location': 'File > Import > Medit Meshes (.mesh)',
    'description': 'Import files in medit mesh format (.mesh)',
    'warning': 'will it work....',
    'wiki_url': 'http://wiki.blender.org/index.php/Extensions:2.5/Py/'\
        'Scripts/',
    'tracker_url': 'https://projects.blender.org/tracker/index.php?'\
        'func=detail&aid=23480',
    'support': 'COMMUNITY',
    'category': 'Import-Export',
    }

import os
import math
from math import sin, cos, radians
import bpy
import mathutils
from mathutils import Vector, Matrix

class OpHelloWorld(bpy.types.Operator):
    bl_idname = "screen.hello_world"
    bl_label = "Hello World"
 
    def execute(self, context):
        self.report({'WARNING'}, "Hello World")
        return {'FINISHED'}
 
# registering and menu integration
def register():
    bpy.utils.register_class(OpHelloWorld)
 
# unregistering and removing menus
def unregister():
    bpy.utils.unregister_class(OpHelloWorld)
 
if __name__ == "__main__":
    register()

# def register():
#     bpy.utils.register_module(__name__)

#     bpy.types.INFO_MT_file_import.append(OpHelloWorld)

 
# def unregister():
#     bpy.utils.unregister_module(__name__)

#     bpy.types.INFO_MT_file_import.remove(OpHelloWorld)


# if __name__ == "__main__":
#     register()
#     print "registering importer"


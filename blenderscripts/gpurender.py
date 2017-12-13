#./blender -b --python gpurender.py trinity2.blend -o ./out --frame-start 1740 --frame-end 1899 -a

import bpy

bpy.context.scene.cycles.device = 'GPU'

#Useful debug info
prefs = bpy.context.user_preferences.addons['cycles'].preferences
print(prefs.compute_device_type)

for d in prefs.devices:
    print(d.name)

#2.76-
#bpy.context.user_preferences.system.compute_device_type = 'CUDA'
#bpy.context.user_preferences.system.compute_device = 'CUDA_1'
#bpy.context.user_preferences.system.compute_device = 'CUDA_MULTI_2' # Uncomment for 2 GPUs
#2.77+
bpy.context.user_preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
bpy.context.user_preferences.addons['cycles'].preferences.compute_device = 'CUDA_1'
#bpy.context.user_preferences.addons['cycles'].preferences.compute_device = 'CUDA_MULTI_2'

bpy.context.scene.render.tile_x = 256
bpy.context.scene.render.tile_y = 256
bpy.context.scene.render.resolution_percentage = 100
bpy.context.scene.render.image_settings.file_format = 'PNG'
print("Rendering on: " + bpy.context.user_preferences.system.compute_device)

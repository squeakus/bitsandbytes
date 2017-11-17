import bpy
import os
import math
import numpy as np
import random
import mathutils
import time
import struct


# Global parameters
USE_GPU = False
PRINT_IMG = True
DIRECTORY = './'


h_res = 320
v_res = 240


def render_frame(i):
    print("processing", i)
    bpy.data.objects['Camera'].select = True

    # Set up rendering of depth map:
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links

    # clear default nodes
    for n in tree.nodes:
        tree.nodes.remove(n)

    # create input render layer node
    rl_node = tree.nodes.new('CompositorNodeRLayers')

    # Create a viewer node for the rgbd image
    view_node_rgb = tree.nodes.new('CompositorNodeViewer')
    view_node_rgb.use_alpha = True

    #Link both the render and view nodes
    links.new(rl_node.outputs[0], view_node_rgb.inputs[0])
    links.new(rl_node.outputs[2], view_node_rgb.inputs[1])

    # Render the scene
    bpy.ops.render.render()

    # Get the image from the view, convert it into numpy array
    pixels = np.array(bpy.data.images['Viewer Node'].pixels)
    print("pixels length", len(pixels))
    print("xres:",bpy.context.scene.render.resolution_x, "yres:", bpy.context.scene.render.resolution_y)
    image_rgbd = pixels[:].reshape(bpy.context.scene.render.resolution_y,bpy.context.scene.render.resolution_x, 4)

    # scaling and inverting for slambench
    depth = image_rgbd[:,:,3] * 10
    depth = np.flipud(depth)
    for row in depth:
        print(row)
    rgb = image_rgbd[:,:,:3]*256
    rgb = np.flipud(rgb)

    if PRINT_IMG:
        np.save(DIRECTORY + 'img_{}'.format(i) ,depth)

    return rgb, depth


def save_slambench(rgb, depth):

    shape = np.ndarray(shape = (2, 1), dtype = np.int32)
    shape[0] = depth.shape[1]
    shape[1] = depth.shape[0]
    print(shape)

    dw = 0
    dw += f.write(shape.astype(np.int32).tobytes())
    dw += f.write(depth[:].astype(np.uint16).tobytes())
    dw += f.write(shape.astype(np.int32).tobytes())
    dw += f.write(rgb[:].astype(np.uint8).tobytes())



""" Setting up camera parameters"""
if (USE_GPU):
    bpy.context.scene.cycles.device = 'GPU'
    prefs = bpy.context.user_preferences.addons['cycles'].preferences
    print(prefs.compute_device_type)

    for d in prefs.devices:
        print(d.name)
    #2.76-
    #bpy.context.user_preferences.system.compute_device_type = 'CUDA'
    #bpy.context.user_preferences.system.compute_device = 'CUDA_1'
    #2.77+
    bpy.context.user_preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
    bpy.context.user_preferences.addons['cycles'].preferences.compute_device = 'CUDA_1'
    #bpy.context.user_preferences.system.compute_device = 'CUDA_MULTI_2' # Uncomment for 2 GPUs

bpy.context.scene.cycles.samples = 500
bpy.context.scene.render.resolution_x = h_res
bpy.context.scene.render.resolution_y = v_res
bpy.context.scene.render.tile_x = 16
bpy.context.scene.render.tile_y = 16
bpy.context.scene.render.resolution_percentage = 100
bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.context.scene.view_settings.gamma = 1
bpy.context.scene.view_settings.exposure = 2

f=open(DIRECTORY +'slambench.raw',"wb")
print("framecount", bpy.context.scene.frame_end)
n_frames = bpy.context.scene.frame_end
for i in range(1,n_frames+1):
    print("rendering frame",i)
    bpy.context.scene.frame_set(frame = i)
    rgb, depth = render_frame(i)
    save_slambench(rgb, depth)

f.close()

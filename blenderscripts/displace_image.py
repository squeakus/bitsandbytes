import bpy 


for o in bpy.data.objects:
    if o.name == "DisplacePlane":
        o.select_set(True)
        bpy.ops.object.delete(use_global=False)


def make_plane(count):
    planename = f"Displaceplane{count:04}"
    texname = f"DisplaceTexture{count:04}"
    imgname = f"img{count:04}.png"
    
    # setting up the displacement plane
    bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
    bpy.context.active_object.name = planename
    bpy.ops.object.shade_smooth()
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.subdivide(number_cuts=300)
    bpy.ops.object.modifier_add(type='DISPLACE')
    plane = bpy.data.objects[plane]

    # load some images for the displacement texture
    bpy.data.images.load(f"//..\\..\\Downloads\\masked\\{imgname}", check_existing=True)
    bpy.data.textures.new(texname, type='IMAGE')

    tex = bpy.data.textures[texname]    
    tex.image = bpy.data.images[imgname]
    plane.modifiers["Displace"].strength = 0.2
    plane.modifiers["Displace"].texture = tex

for i in range(10):
    make_plane()

bpy.ops.object.mode_set(mode='OBJECT')


#tex.keyframe_insert(data_path=filepath, frame=1)
#tex.name
#bpy.data.images["img0001.png"].filepath = "//..\\..\\Downloads\\masked\\img0002.png"
#tex.keyframe_insert(data_path=filepath, frame=10)
#bpy.data.images["img0001.png"].filepath = "//..\\..\\Downloads\\masked\\img0003.png"
#tex.keyframe_insert(filepath, frame=20)

#

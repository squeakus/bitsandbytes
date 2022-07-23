import bpy 


def clean_scene():
    # first delete everything
    for o in bpy.data.objects:
        if o.name.startswith("DisplacePlane"):
            print(o.name)
            o.select_set(True)
            if o.mode == 'EDIT':
                o.mode ='OBJECT'
            bpy.ops.object.delete(use_global=False)

    for block in bpy.data.textures:
        if block.users == 0:
            bpy.data.textures.remove(block)

def make_plane(count):
    planename = f"DisplacePlane{count:04}"
    texname = f"DisplaceTexture{count:04}"
    imgname = f"img{count:04}.png"
    
    # setting up the displacement plane
    bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
    bpy.context.active_object.name = planename
    bpy.ops.object.shade_smooth()
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.subdivide(number_cuts=300)
    bpy.ops.object.modifier_add(type='DISPLACE')
    plane = bpy.data.objects[planename]

    # load some images for the displacement texture
    bpy.data.images.load(f"//..\\..\\Downloads\\masked\\{imgname}", check_existing=True)
    if texname not in bpy.data.textures.keys():
        bpy.data.textures.new(texname, type='IMAGE')

    tex = bpy.data.textures[texname]    
    tex.image = bpy.data.images[imgname]
    plane.modifiers["Displace"].strength = 0.2
    plane.modifiers["Displace"].texture = tex
    bpy.ops.object.mode_set(mode='OBJECT')
    plane.hide_render = False

clean_scene()

for i in range(10):
    make_plane(i)




#tex.keyframe_insert(data_path=filepath, frame=1)
#tex.name
#bpy.data.images["img0001.png"].filepath = "//..\\..\\Downloads\\masked\\img0002.png"
#tex.keyframe_insert(data_path=filepath, frame=10)
#bpy.data.images["img0001.png"].filepath = "//..\\..\\Downloads\\masked\\img0003.png"
#tex.keyframe_insert(filepath, frame=20)

#

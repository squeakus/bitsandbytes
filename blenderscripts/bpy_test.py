import bpy 


for o in bpy.context.scene.objects:
    if o.name == "DisplacePlane":
        if bpy.context.object.mode == 'EDIT':
            bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='DESELECT')
        bpy.data.objects['DisplacePlane'].select_set(True)
        bpy.ops.object.delete(use_global=False)

# setting up the displacement plane
bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
bpy.context.active_object.name = 'DisplacePlane'
bpy.ops.object.shade_smooth()
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.subdivide(number_cuts=300)
bpy.ops.object.modifier_add(type='DISPLACE')
plane = bpy.data.objects['DisplacePlane']

# load some images for the displacement texture
bpy.data.images.load("//..\\..\\Downloads\\masked\\img0001.png", check_existing=True)
bpy.data.images.load("//..\\..\\Downloads\\masked\\img0002.png", check_existing=True)
bpy.data.images.load("//..\\..\\Downloads\\masked\\img0003.png", check_existing=True)

if not "DisplaceTexture" in bpy.data.textures.keys():
    bpy.data.textures.new("DisplaceTexture", type='IMAGE')

tex = bpy.data.textures["DisplaceTexture"]    
tex.image = bpy.data.images['img0001.png']
plane.modifiers["Displace"].strength = 0.2
plane.modifiers["Displace"].texture = tex

tex.keyframe_insert(data_path=filepath, frame=1)
tex.name
bpy.data.images["img0001.png"].filepath = "//..\\..\\Downloads\\masked\\img0002.png"
tex.keyframe_insert(data_path=filepath, frame=10)
bpy.data.images["img0001.png"].filepath = "//..\\..\\Downloads\\masked\\img0003.png"
tex.keyframe_insert(filepath, frame=20)

bpy.ops.object.mode_set(mode='OBJECT')

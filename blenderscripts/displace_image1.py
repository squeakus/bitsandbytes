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

    # order is important
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)

    for block in bpy.data.textures:
        if block.users == 0:
            bpy.data.textures.remove(block)

def make_plane(count):
    planename = f"DisplacePlane{count:04}"
    texname = f"DisplaceTexture{count:04}"
    imgname = f"img{count:04}.png"
    matname = f"material{count:04}"
        
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
    
    mat = bpy.data.materials.get(matname)
    if mat is None:
        # create material
        mat = bpy.data.materials.new(name=matname)
    mat.use_nodes = True
    image_node = mat.node_tree.nodes.new('ShaderNodeTexImage')
    image_node.image = bpy.data.images[imgname]
    bsdf_node = mat.node_tree.nodes["Principled BSDF"]
    links = mat.node_tree.links.new(image_node.outputs["Color"], bsdf_node.inputs["Base Color"])
    plane.active_material = mat
    
    plane.modifiers["Displace"].strength = 0.2
    plane.modifiers["Displace"].texture = tex
    bpy.ops.object.mode_set(mode='OBJECT')
    plane.hide_render = True
    plane.keyframe_insert(data_path="hide_render", frame=1)

clean_scene()
image_count = 10
for i in range(image_count):
    make_plane(i)

plane = bpy.data.objects["DisplacePlane0000"]
plane.hide_render = False
plane.keyframe_insert(data_path="hide_render", frame=1)
plane.hide_render = True
plane.keyframe_insert(data_path="hide_render", frame=2)

     
for i in range(image_count-1):
    prev_planename = f"DisplacePlane{i:04}"
    next_planename = f"DisplacePlane{i+1:04}"
    prev_plane = bpy.data.objects[prev_planename]
    next_plane = bpy.data.objects[next_planename]
    prev_plane.hide_render = True
    next_plane.hide_render = False
    prev_plane.keyframe_insert(data_path="hide_render", frame=i)
    next_plane.keyframe_insert(data_path="hide_render", frame=i)


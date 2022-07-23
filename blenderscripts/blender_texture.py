import bpy

bpy.ops.object.delete(use_global=False)
bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
bpy.ops.object.editmode_toggle()
bpy.ops.mesh.subdivide(number_cuts=100)
bpy.context.space_data.context = 'MODIFIER'
bpy.ops.object.modifier_add(type='DISPLACE')
bpy.ops.texture.new()
bpy.context.space_data.context = 'TEXTURE'
bpy.context.space_data.system_bookmarks_active = 3
bpy.ops.image.open(filepath="C:\\Users\\Jonathan\\Downloads\\mountfocus.png", directory="C:\\Users\\Jonathan\\Downloads\\", files=[{"name":"mountfocus.png", "name":"mountfocus.png"}], relative_path=True, show_multiview=False)
bpy.ops.object.editmode_toggle()
bpy.ops.object.modifier_add(type='DISPLACE')
bpy.context.space_data.context = 'MODIFIER'
bpy.context.object.modifiers["Displace"].strength = 0.3
bpy.ops.object.modifier_apply(modifier="Displace")
bpy.context.space_data.context = 'MATERIAL'
bpy.ops.material.new()
bpy.context.object.active_material.use_nodes = True
bpy.ops.object.shade_smooth()
bpy.context.space_data.shading.show_backface_culling = True
bpy.context.space_data.shading.show_cavity = True
bpy.context.area.ui_type = 'ShaderNodeTree'
bpy.ops.node.select(wait_to_deselect_others=False, mouse_x=34, mouse_y=574, deselect_all=True)
bpy.ops.node.add_search(use_transform=True, node_item='42')
bpy.ops.node.translate_attach_remove_on_cancel(TRANSFORM_OT_translate={"value":(-141.697, -20.7614, 0), "orient_axis_ortho":'X', "orient_type":'GLOBAL', "orient_matrix":((1, 0, 0), (0, 1, 0), (0, 0, 1)), "orient_matrix_type":'GLOBAL', "constraint_axis":(False, False, False), "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_target":'CLOSEST', "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":True, "view2d_edge_pan":True, "release_confirm":False, "use_accurate":False, "use_automerge_and_split":False}, NODE_OT_attach={}, NODE_OT_insert_offset={})
bpy.context.space_data.system_bookmarks_active = 3
bpy.ops.image.open(filepath="C:\\Users\\Jonathan\\Downloads\\mountfocus.png", directory="C:\\Users\\Jonathan\\Downloads\\", files=[{"name":"mountfocus.png", "name":"mountfocus.png"}], show_multiview=False)
bpy.ops.node.select(wait_to_deselect_others=True, mouse_x=133, mouse_y=619, deselect_all=True)
bpy.ops.node.link(detach=False, has_link_picked=False, drag_start=(-126.543, 298.466))
bpy.context.space_data.shading.type = 'MATERIAL'
bpy.context.area.ui_type = 'CONSOLE'
bpy.context.area.ui_type = 'INFO'
bpy.ops.outliner.item_activate(deselect_all=True)
bpy.ops.anim.keyframe_insert()
bpy.ops.transform.translate(value=(-2.88247, -0, -0), orient_axis_ortho='X', orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(True, False, False), mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)
bpy.ops.transform.translate(value=(0, 2.70277, 0), orient_axis_ortho='X', orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(False, True, False), mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)
bpy.ops.transform.translate(value=(-0, -0, -2.05292), orient_axis_ortho='X', orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(False, False, True), mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)
bpy.context.scene.tool_settings.use_keyframe_insert_auto = True
bpy.ops.transform.translate(value=(-3.26826, -0, -0), orient_axis_ortho='X', orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(True, False, False), mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)
bpy.ops.transform.translate(value=(0, 3.40007, 0), orient_axis_ortho='X', orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(False, True, False), mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)
bpy.ops.transform.translate(value=(-0, -0, -2.21544), orient_axis_ortho='X', orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(False, False, True), mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)
bpy.ops.transform.translate(value=(-0.334393, -0.0723012, 1.28523e-07), orient_axis_ortho='X', orient_type='VIEW', orient_matrix=((0.685921, 0.727676, 1.49012e-08), (-0.324013, 0.305421, 0.895396), (0.651558, -0.61417, 0.445271)), orient_matrix_type='VIEW', mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)
bpy.ops.transform.translate(value=(0, 2.97691, 0), orient_axis_ortho='X', orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(False, True, False), mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)
bpy.ops.transform.rotate(value=0.736997, orient_axis='Z', orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(False, False, True), mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)
bpy.ops.transform.translate(value=(0.745608, -0.605523, 1.11759e-06), orient_axis_ortho='X', orient_type='VIEW', orient_matrix=((0.0188698, 0.999822, -0), (-0.445192, 0.00840218, 0.895396), (0.895236, -0.0168959, 0.445271)), orient_matrix_type='VIEW', mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)
bpy.ops.transform.rotate(value=-0.00503769, orient_axis='Z', orient_type='VIEW', orient_matrix=((0.0188698, 0.999822, -0), (-0.445192, 0.00840218, 0.895396), (0.895236, -0.0168959, 0.445271)), orient_matrix_type='VIEW', mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)
bpy.ops.transform.translate(value=(-0.453634, -0, -0), orient_axis_ortho='X', orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(True, False, False), mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)
bpy.ops.transform.translate(value=(-0, -0, -0.44234), orient_axis_ortho='X', orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(False, False, True), mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)

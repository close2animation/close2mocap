import bpy


def delete_unchanging_fcurves(action1, action2):
    data_paths = []
    for fc1, fc2 in zip(action1.fcurves, action2.fcurves):
        if fc1.evaluate(0) == fc2.evaluate(0):
            action1.fcurves.remove(fc1)
            action2.fcurves.remove(fc2)
        else:
            data_paths.append(fc1.data_path + str(fc1.array_index))
    return data_paths


def delete_data_paths_from_action(data_paths, action):
    for fc in action.fcurves:
        dp = fc.data_path + str(fc.array_index)
        if not dp in data_paths:
            action.fcurves.remove(fc)


def select_asset_in_browser(context, action, deselect=False):
    ''' select an asset in the asset browser '''
    for space in context.spaces:
        if hasattr(space, 'activate_asset_by_id'):
            space.browse_mode = 'ASSETS'
            space.activate_asset_by_id(action)
            if deselect:
                space.deselect_all()


def blend_between_poses(blend_factor, action, action1, arm):
    ''' apply a blend between two poses '''
    old_area = bpy.context.area.type
    arm.pose.apply_pose_from_action(action) 
    override = bpy.context.copy()
    override['area'].type = 'FILE_BROWSER'
    #select_asset_in_browser(override['area'], action1)
    bpy.ops.poselib.blend_pose_asset(override, blend_factor=blend_factor)
    bpy.ops.poselib.pose_asset_select_bones(override)
    bpy.context.area.type = old_area


def keyframe_selected_bones(frame):
    bones = bpy.context.selected_pose_bones
    for bone in bones:
        print(bone.name)
        if bone.rotation_mode == 'QUATERNION':
            rot_mode = 'rotation_quaternion'
        else:
            rot_mode = 'rotation_euler'
        print(bone)    
        bone.keyframe_insert(data_path='location', frame=frame)
        bone.keyframe_insert(data_path=rot_mode, frame=frame)


def add_track_with_action(animation_data, action):
    track = animation_data.nla_tracks.new()
    track.name = action.name
    strip = track.strips.new(action.name, 1, action)
    strip.blend_type = 'ADD'

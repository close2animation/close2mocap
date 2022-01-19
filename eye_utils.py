from numpy.lib.function_base import insert
import bpy
import numpy as np
import cv2
import mathutils


def get_mag(v1, v2, flat=False):
    ''' returns the mag between two vectors'''
    if flat:
        v1 = v1[:, :2]
        v2 = v2[:, :2]
        
    v3 = (v1 - v2)
    v3 = v3 ** 2
    mag = np.sqrt(np.sum(v3, axis=1))
    print('mag', mag[0])
    
    return mag


def norm_arr(distance, scale):
    '''normalise range'''
    # scale distance 
    distance = distance / scale 
    max = np.max(distance)
    min = np.min(distance)
    distance = (distance - min) /(max - min)
    return distance


def create_empty(name, data):
    '''create empty with mocap data'''
    bpy.ops.mesh.primitive_cube_add()   
    bpy.context.active_object.name = name
    obj = bpy.context.active_object
    for i, d in enumerate(data):
        obj.location[0] = d
        obj.keyframe_insert(data_path="location", frame=i)


def nothing(x):
    pass


def get_shape(landmarks, verts_idx, mask, kx, ky):
    ''' use the face verts to create a shape for the mask'''
    points = [landmarks[idx] for idx in verts_idx]
    points = np.array(points)[:, :2]
    points[:, 0] *= mask.shape[1]
    points[:, 1] *= mask.shape[0]
    points = points.astype(int)
    mask = cv2.fillConvexPoly(mask, points, 255)
    
    kernel = np.ones((kx, ky), np.uint8)
    mask = cv2.erode(mask, kernel)
 
    return mask

def find_eye_centre(contours, idx):
    ''' finds the centre of the contour'''
    x = 0
    y = 0
    print(contours[idx])
    points = len(contours[idx])
    for kp in contours[idx]:
        x = x+kp[0][0]
        y = y+kp[0][1]    
    x = int(x/points)
    y = int(y/points)
    return (x, y)


def find_iris(img, img_o):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    if len(contours) > 0:
        #draw cirlce on eye
        eye_loc = find_eye_centre(contours, 0)
        cv2.circle(img_o, (eye_loc[0], eye_loc[1]), 1, (0, 0, 255), 2)
    else:
        eye_loc = (0, 0)
    
    return eye_loc 


def collect_data_from_pose_library(action, transform_type, side):
    # lol what a waste of time
    ''' returns pose library action into an easier format to work with

        shape ---> 'bone name' : [[[pose data frame 1][array_indexs]], 
                                  [[pose data frame 2][array_indexs]]]
    '''
    
    action_eye = bpy.data.actions['eyes']
    open_marker_eye = action_eye.pose_markers['eyes_open'].frame 
    closed_marker_eye = action_eye.pose_markers['eyes_closed'].frame
    pose_data = {}
    data_open = []
    idx_open = []
    data_closed = []
    idx_closed = []
    data_path_current = ''
    for fcurve in action.fcurves:
        # only loops over transform type
        if fcurve.data_path.split('.')[-1] != transform_type:
            continue
        # separating left and right 
        if fcurve.data_path.split('.')[-2][0] != side:
            continue
        # ignore if the starting and resulting transform is the same
        if fcurve.evaluate(open_marker_eye) == fcurve.evaluate(closed_marker_eye):
            continue
        # create data
        if fcurve.data_path != data_path_current:
            pose_data[data_path_current] = [[data_open, idx_open],
                                            [data_closed, idx_closed]] 
            
            data_path_current = fcurve.data_path                                 
            # create new list open
            data_open = []
            idx_open = []
            data_open.append(fcurve.evaluate(open_marker_eye))
            idx_open.append(fcurve.array_index)
           
            # create new list close
            data_closed = []
            idx_closed = []
            data_closed.append(fcurve.evaluate(closed_marker_eye))
            idx_closed.append(fcurve.array_index)
            
        else:
            data_open.append(fcurve.evaluate(open_marker_eye))
            idx_open.append(fcurve.array_index)
            data_closed.append(fcurve.evaluate(closed_marker_eye))
            idx_closed.append(fcurve.array_index)

    pose_data[data_path_current] = [[data_open, idx_open],
                                    [data_closed, idx_closed]]
    del pose_data['']
    return pose_data


def set_interpolation_location(pose_data, scale, obj, frame):
    for key, value in pose_data.items():
        start = mathutils.Vector(value[0][0])
        end = mathutils.Vector(value[1][0])
        mid = start.lerp(end, scale)
        bone_name = key.split('"')[1]
        print('bone_name', bone_name)
        obj.pose.bones[bone_name].location = mid 
        obj.pose.bones[bone_name].keyframe_insert(data_path='location', frame=frame)


def set_interpolation_rotation_q(pose_data, scale, obj, frame):
    for key, value in pose_data.items():
        print(key)
        start = mathutils.Quaternion(value[0][0])
        end = mathutils.Quaternion(value[1][0])
        mid = start.slerp(end, scale)
        bone_name = key.split('"')[1]
        print('bone_name', bone_name)
        obj.pose.bones[bone_name].rotation_quaternion = mid
        obj.pose.bones[bone_name].keyframe_insert(data_path='rotation_quaternion', frame=frame)


def collect_group_data(group1, group2):
    # test cause i dont trust
    if not(group1.name == group2.name):
        print('shhhhhhhhhhhhhhhhhhit that aint right')
        return None
    
    location_1 = []
    location_2 = []
    location_idx = []
    rotation_1 = []
    rotation_2 = []
    rotation_idx = []
    for fcurve1, fcurve2 in zip(group1.channels, group2.channels):
        if fcurve1.data_path.split('.')[-1] == 'location':
            location_1.append(fcurve1.evaluate(0)) 
            location_2.append(fcurve2.evaluate(0))
            location_idx.append(fcurve1.array_index)
            
        elif fcurve1.data_path.split('.')[-1] == 'rotation_quaternion':
            rotation_1.append(fcurve1.evaluate(0)) 
            rotation_2.append(fcurve2.evaluate(0))
            rotation_idx.append(fcurve1.array_index)
            
        elif fcurve1.data_path.split('.')[-1] == 'rotation_euler':
            rotation_1.append(fcurve1.evaluate(0)) 
            rotation_2.append(fcurve2.evaluate(0))
            rotation_idx.append(fcurve1.array_index)
    return (location_1, location_2), (rotation_1, rotation_2), location_idx


def apply_lerp_to_bone_loc(loc, scale, bone_name, obj, frame, location_idx, add=None):
    # lerp_location     
    if len(loc[0]) > 1:
        loc1 = mathutils.Vector(loc[0])
        loc2 = mathutils.Vector(loc[1])
        loc_lerp = loc1.lerp(loc2, scale) 
        if add != None:
            loc_lerp = loc_lerp + add

        if len(loc[0]) == 3:
            obj.pose.bones[bone_name].location = loc_lerp 
            obj.pose.bones[bone_name].keyframe_insert(data_path="location", frame=frame)
            return loc_lerp
        else:
            obj.pose.bones[bone_name].location[location_idx[0]] = loc_lerp[0]
            obj.pose.bones[bone_name].location[location_idx[1]] = loc_lerp[1]
            obj.pose.bones[bone_name].keyframe_insert(data_path="location", frame=frame)
            return loc_lerp
                       
    else:
        range = loc[1] - loc[0]
        loc_lerp = range * scale
        loc_lerp += loc[0]
        obj.pose.bones[bone_name].location = loc_lerp
        obj.pose.bones[bone_name].keyframe_insert(data_path="location", frame=frame)
        return loc_lerp


def apply_lerp_to_bone_rot(rot, scale, bone_name, obj, frame, add=None):
    # lerp rotation
    if len(rot[0]) == 3:
        rot1 = mathutils.Euler(rot[0], 'XYZ').to_quaternion()
        rot2 = mathutils.Euler(rot[1], 'XYZ').to_quaternion()
        rot_lerp = rot1.slerp(rot2, scale)
        rot_lerp = rot_lerp.to_euler()
        if add != None:
            xyz = (rot_lerp[0] + add[0]), (rot_lerp[1] + add[1]), (rot_lerp[2] + add[2])
            rot_lerp = mathutils.Vector(xyz)
        obj.pose.bones[bone_name].rotation_euler = rot_lerp
        obj.pose.bones[bone_name].keyframe_insert(data_path="rotation_euler", frame=frame)
        return rot_lerp
   
    elif len(rot[0]) == 4:
        rot1 = mathutils.Quaternion(rot[0])
        rot2 = mathutils.Quaternion(rot[1])
        rot_lerp = rot1.slerp(rot2, scale)
        if add != None:
            rot_lerp = rot_lerp + add
        obj.pose.bones[bone_name].rotation_quaternion = rot_lerp 
        obj.pose.bones[bone_name].keyframe_insert(data_path="rotation_quaternion", frame=frame)
        return rot_lerp

         
def lerp_between_poses(action_groups1, action_groups2, scale, obj, frame, side=None, change=None):  
    # wow i really fuked this up. w/e another thing that'll need to be rewriten.
    # loops through the bones
    loc_list = []
    rot_list = []

    if change != None:
        change_l = iter(change[0])
        change_r = iter(change[1])
    else:
        add_loc = None
        add_rot = None

    for idx, (group1, group2) in enumerate(zip(action_groups1, action_groups2)):
        loc, rot, location_idx = collect_group_data(group1, group2)
        if group1.name.split('.')[-1] == side:
            continue

        # if there's no change between poses, ignore.
        if loc[0] != loc[1]:
            if change != None:
                add_loc = next(change_l)
            loc_final = apply_lerp_to_bone_loc(loc, scale, group1.name, obj, frame, location_idx, add_loc)
            loc_list.append(loc_final)

        if rot[0] != rot[1]:
            if change != None:
                add_rot = next(change_r)
            rot_final = apply_lerp_to_bone_rot(rot, scale, group1.name, obj, frame, add_rot)
            rot_list.append(rot_final)

    return (loc_list, rot_list)         

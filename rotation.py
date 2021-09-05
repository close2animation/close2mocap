import bpy
import bmesh
from mathutils import *; from math import *
import numpy as np
import math


def adjust_point_cloud_loc(face_mesh_list, pin_idx, ignore=None):
    ''' will change this later'''

    new_face_mesh_list = []
    for face_mesh in face_mesh_list:
        pin_point = face_mesh[pin_idx]
        new_face_mesh = []
        for idx, vert in enumerate(face_mesh):
            if ignore == None:                
                vert = vert - pin_point  # cant do -= here
                new_face_mesh.append(vert)
            else:
                for i in range(len(ignore)):
                    if idx == ignore[i]:
                        pass
                    else:
                        vert[0] = vert[0] - pin_point[0]
                        vert[2] = vert[2] - pin_point[2]  # cant do -= here
                        new_face_mesh.append(vert)
        new_face_mesh_list.append(new_face_mesh)
    return new_face_mesh_list


def rotate_along_axis(verts, axis, main_point_idx, target):
    # remove nil axis
    verts = np.delete(verts, axis, axis=1)

    # caluclate radius of each vert
    verts_squared = verts**2
    radius = np.sqrt(np.sum(verts_squared, axis=1))
    
    # define main point
    mp_radius = radius[main_point_idx]
    mp = verts[main_point_idx]
    
    # get angles relative to main point
    dot = np.dot(verts, mp)
    mag = radius * mp_radius
    angle = np.arccos(dot/mag)
    
    # get angles relative to target.
    dot = np.dot(verts, target)
    mag = radius * target[0]
    angles = np.arccos(dot/mag)

    # finding out if points are left or right of main point
    slope = mp[1] / mp[0] 
    if mp[0] > 0: 
        side = np.argwhere(verts[:,1] >= verts[:,0] * slope)
    else:
        side = np.argwhere(verts[:,1] <= verts[:,0] * slope)

    # adjusting angle offset 
    for i in side:
        angle[i] = np.negative(angle[i])     
    offset = np.radians(90) 
    angle += offset
    angle[main_point_idx] = np.radians(90)
    
    # creating x,y vaules
    y = radius *  np.cos(angle)
    x = radius *  np.sin(angle)
    verts = np.stack((x,y), axis=1)
    
    # return verts and amount the object was rotated
    return verts, angles[main_point_idx]


def align_to_target_axis(verts, target, main_point, twist_point):
    '''
    rotates an object so that the choosen vert aligns to the x axis.
    also records amount of rotation needed to get there.
    '''
    # facing z
    verts_transformed, rotation_z = rotate_along_axis(verts, 2, main_point, target)
    verts[:, 0] = verts_transformed[:, 0]
    verts[:, 1] = verts_transformed[:, 1]
    
    # facing y
    verts_transformed, rotation_y = rotate_along_axis(verts, 1, main_point, target)
    verts[:, 0] = verts_transformed[:, 0]
    verts[:, 2] = verts_transformed[:, 1]

    # facing x
    verts_transformed, rotation_x = rotate_along_axis(verts, 0, twist_point, target)
    verts[:, 1] = verts_transformed[:, 0]
    verts[:, 2] = verts_transformed[:, 1]
    return verts  


def get_specific_fcurves(obj, transform_type, axis):
    '''
    take an obj and returns a specific transform
    '''
    axes = {'x' : [0], 'y' : [1], 'z' : [2], 'all' : [0, 1, 2]}
    list = []
        
    for fcurve in obj.animation_data.action.fcurves:
        if fcurve.data_path == transform_type and fcurve.array_index in axes[axis]:
            list.append(fcurve)
    return list


def fcurve_to_list(fcurve):
    keypoints = []
    for frame in fcurve.keyframe_points:
        keypoints.append(frame.co[1])
    return keypoints
      
              
def collect_data_in_collection(name):
    data = []
    for obj in bpy.data.collections[name].objects:
        fcurves = get_specific_fcurves(obj, 'location', 'all')
        location = []
        for fcurve in fcurves:
            location.append(fcurve_to_list(fcurve))
        data.append(location)
    return data


def move_to_world_origin(obj, idx):
    '''
    takes a list of objects and slides the choosen vert to 0,0,0
    '''
    obj -= np.reshape(obj[:, idx], (obj.shape[0], 1, obj.shape[2]))
    return obj


def find_joint_angle(finger, joints):
    target = np.array([1,0])
      
    # set origin
    #data_hand = move_to_world_origin(data_hand, joint1)
    finger -= finger[joints[0]] 
    
    # algin joint along x axis
    finger = align_to_target_axis(finger, target, joints[1], joints[2])

    # doing this since origin becomes nan 
    finger[joints[0]] = [0,0,0]

    # slide joint down
    finger -= finger[joints[1]] 
    #create_object(finger, 'result')

    # find angle between joint3 and target
    angle_point = finger[joints[2]][:2]
    angle = math.atan2(angle_point[1], angle_point[0])
    
    #angle = np.arctanh(angle_point[1]/ angle_point[0])
    #print(math.degrees(angle))
    return angle
    

def get_finger_angles(finger):
    finger_angles = []
    joints = [0, 1, 2]
    angle = find_joint_angle(finger, joints)
    finger_angles.append(angle)
    joints = [1, 2, 3]
    angle = find_joint_angle(finger, joints)
    finger_angles.append(angle)
    joints = [2, 3, 4]
    angle = find_joint_angle(finger, joints)
    finger_angles.append(angle)
    return finger_angles


def add_animation_to_finger(finger_angles, label, offset):
    joint1 = finger_angles[:, 0]
    joint2 = finger_angles[:, 1]
    joint3 = finger_angles[:, 2]
    obj = bpy.context.scene.target
    
    for frame in range(len(finger_angles)):  
        #print(angles)
        obj.pose.bones[label + '.01.R'].rotation_mode = 'XYZ'
        obj.pose.bones[label + '.01.R'].rotation_euler[0] = joint1[frame] + offset[0]
        obj.pose.bones[label + '.01.R'].keyframe_insert(data_path="rotation_euler", frame=frame)
        
        obj.pose.bones[label + '.02.R'].rotation_mode = 'XYZ'
        obj.pose.bones[label + '.02.R'].rotation_euler[0] = joint2[frame] + offset[0]
        obj.pose.bones[label + '.02.R'].keyframe_insert(data_path="rotation_euler", frame=frame)
        
        obj.pose.bones[label + '.03.R'].rotation_mode = 'XYZ'
        obj.pose.bones[label + '.03.R'].rotation_euler[0] = joint3[frame] + offset[0]
        obj.pose.bones[label + '.03.R'].keyframe_insert(data_path="rotation_euler", frame=frame)


def select_finger(data_hand, idx1, idx2):
    # testing a single finger
    finger1 = data_hand[:, idx1:idx2]
    base = np.reshape(data_hand[:, 0], (data_hand.shape[0], 1, 3))
    finger = np.concatenate([base, finger1], axis=1)

    # get angles
    finger_angles = []
    for frame in finger:
        angles = get_finger_angles(frame)
        finger_angles.append(angles)
    
    return finger_angles
    
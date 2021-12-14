import numpy as np
import bpy
from mathutils import Matrix, Vector
import bmesh
import mathutils
import copy


def normalise_vector_batch(vector):
    '''
    normalising vectors so that (a**2 + b**2 + c**2 == 1**2) 
    '''
    vector_s = vector ** 2
    vector_s = np.sum(vector_s, axis=1)
    vector_s = np.sqrt(vector_s)
    vector_s = vector_s.reshape(vector_s.shape[0], 1)
    
    return vector / vector_s


def axis_angle_to_quaternion(angle, vector):
    '''
    converts axis_angle (theta, vector) to quaternion (w, x, y, z). 
    formula for defining (w, x, y, z) --> (cos(theta/2), v1*sin(theta/2), v2*sin(theta/2), v3*sin(theta/2)) 
    '''
    w = np.cos(angle/2)
    xyz = np.sin(angle/2) * vector
    quat = np.hstack((w,xyz))
    return quat


def angle_between_two_vectors_batch(vec1, vec2):
    dot = np.dot(vec1, vec2)
    # rounding since the dot product of same vector returns value over 1
    dot = np.round(dot, 5)
    angle = np.arccos(dot/1)
    return angle


def generate_quaternion_from_3d_point(points, target):
    points = normalise_vector_batch(points)

    # create axis vector
    axis_vector = np.cross(points, target)
    axis_vector = normalise_vector_batch(axis_vector)

    # create angle                
    angles = angle_between_two_vectors_batch(points, target)
    angles = angles.reshape(angles.shape[0], 1)
    angles = -angles
        
    # create quaternion 
    quat = axis_angle_to_quaternion(angles, axis_vector)           
    return quat


def quats_to_rotation_mats(quats, invert=False):
    ''' input numpy '''
    
    R_list = []
    for quat in quats:
        R = mathutils.Quaternion(quat)
        R = R.to_matrix()
        if invert:
            R.invert()
        R_list.append(R)
    return np.array(R_list)


def rotate_batch(points, R_batch):
    ''' input numpy '''
    point_list = []
    for point, R in zip(points, R_batch):
        p1  = np.matmul(point, R)
        point_list.append(p1)
        
    return np.array(point_list)


def rotate_points(points, R):
    ''' input numpy '''
    point_list = []
    for point in points:
        p1  = np.matmul(point, R)
        point_list.append(p1)
        
    return np.array(point_list)


def generate_quaternion_batch(p1, p2):
    ''' generate quaternion where point1 defines up and point2 defines object's twist '''
    
    # generate first rotation
    target = np.array([0, 0, 1])
    quats = generate_quaternion_from_3d_point(p1, target)

    # shifting verts for second point so object points up and z = 0
    R_batch = quats_to_rotation_mats(quats)
    p2 = rotate_batch(p2, R_batch)
    p2[:, 2] = 0

    # generate second rotation
    target = np.array([1, 0, 0])
    quats2 = generate_quaternion_from_3d_point(p2, target)
    
    # merge rotations
    final_quats = []
    for q1, q2 in zip(quats, quats2):
        turn = mathutils.Quaternion(q1)
        twist = mathutils.Quaternion(q2)
        final_quat = turn @ twist
        final_quats.append(final_quat)
    
    return np.array(final_quats)


def create_mesh_animation(animation_data, object_name):
    ''' input shape ---> frames/vert/coor '''
    
    animation_data = np.transpose(animation_data, (1, 0, 2))
    
    obj = bpy.data.objects[object_name]
    mesh = obj.data
    action = bpy.data.actions.new("meshAnimation")
    mesh.animation_data_create()
    mesh.animation_data.action = action
    
    # loop over verts
    for idx, vert in enumerate(animation_data):
        # create fcurves for vert (xyz)
        fcurves = [action.fcurves.new(f'vertices[{idx}].co', index=i) for i in range(3)]  
        for frame, frame_data in enumerate(vert):    
            fcurves[0].keyframe_points.insert(frame, frame_data[0], options={'FAST'}) # x
            fcurves[1].keyframe_points.insert(frame, frame_data[1], options={'FAST'}) # y
            fcurves[2].keyframe_points.insert(frame, frame_data[2], options={'FAST'}) # z


def offset_quat_by_first_index(quats):
    # offset rotation
    offsets = []
    for quat in quats:
        offset = mathutils.Quaternion(quat) @ mathutils.Quaternion(quats[0]).inverted()
        offsets.append(offset)
        
    return offsets
        
        
def offset_quat(quats, offset):
    change = copy.copy(offset)
    # offset rotation
    change = mathutils.Quaternion(change).inverted()
    offsets = []
    for quat in quats:
        offset = mathutils.Quaternion(quat) @ change
        offsets.append(offset)
        
    return np.array(offsets)


def apply_rotation_to_rig_quat(rotation, obj, inverse=True):
    # apply to test model
    obj.pose.bones['head'].rotation_mode = 'QUATERNION'
    for i, r in enumerate(rotation):
        if inverse:
            r = mathutils.Quaternion(r).inverted()
        obj.pose.bones['head'].rotation_quaternion = r
        obj.pose.bones['head'].keyframe_insert(data_path="rotation_quaternion", frame=i)

# -----------------------------------------------------------------------------------------------------
# clean this 

def apply_quaternion_to_bone(rotation, obj, bone_name, inverse=False):
    # apply to test model
    obj.pose.bones[bone_name].rotation_mode = 'QUATERNION'
    for i, r in enumerate(rotation):
        if inverse:
            r = mathutils.Quaternion(r).inverted()
            
        print('r')
        print(r)
        obj.pose.bones[bone_name].rotation_quaternion = r
        obj.pose.bones[bone_name].keyframe_insert(data_path="rotation_quaternion", frame=i)
        

def add_rotation_q_to_bone(rotation, obj, bone_name, inverse=False):
    # apply to test model
    obj.pose.bones[bone_name].rotation_mode = 'QUATERNION'
    for i, r in enumerate(rotation):
        if inverse:
            r = mathutils.Quaternion(r).inverted()
        bpy.context.scene.frame_set(i)
        r_initial = obj.pose.bones[bone_name].rotation_quaternion
        
        print('r_initial')
        print(r_initial)
        
        obj.pose.bones[bone_name].rotation_quaternion = r_initial @ mathutils.Quaternion(r)
        obj.pose.bones[bone_name].keyframe_insert(data_path="rotation_quaternion", frame=i)


def slope_angle_2d(p1, p2, target):
    # p1 to origain
    points = p2 - p1
    points = np.concatenate((points, np.zeros((points.shape[0], 1))), axis=1)
    quats = generate_quaternion_from_3d_point(points, target)
    return quats


def get_middle_point_2d(point2, point1):
    # finding midpoint between 1 and 2
    x = point2[:, 0] - point1[:, 0]  
    x = (x*.5) + point1[:, 0] 
    y = point2[:, 1] - point1[:, 1]  
    y = (y*.5) + point1[:, 1]
    
    # merge
    x = x.reshape(x.shape[0], 1)
    y = y.reshape(y.shape[0], 1)
    xy = np.concatenate((x, y), axis=1)
    
    return xy

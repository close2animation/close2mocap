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

   



import bpy
import numpy as np
import cv2


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


def get_shape(landmarks, verts_idx, mask):
    ''' use the face verts to create a shape for the mask'''
    points = [landmarks[idx] for idx in verts_idx]
    mask = cv2.fillConvexPoly(mask, np.array(points), 255)

    #kernel = np.ones((7, 3), np.uint8)
    #mask = cv2.erode(mask, kernel)
 
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
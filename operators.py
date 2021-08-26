import bpy
from pathlib import Path
from bpy_extras.io_utils import ImportHelper 
from bpy.types import Operator, OperatorFileListElement
from bpy.props import CollectionProperty, StringProperty
import mediapipe as mp
import cv2
import numpy as np
from .head_rotation import *
import os
from .distance_utils import  *



def create_folder(dir, folder_name):
    ''' just creates a folder and returns the directory'''
    path = dir + folder_name
    try:
        os.makedirs(path)
    except FileExistsError:
        # directory already exists
        print('directory already exists')
        pass
    return path 


def smooth_curve(object_name, transform_type='location', axis=2, kernel_size=10):
    '''applys a simple smoothing average to a f-curve'''

    obj = bpy.data.objects[object_name]

    # refernce axis and data path (eg. location)
    for fcurve in obj.animation_data.action.fcurves:
        if fcurve.data_path == transform_type and fcurve.array_index == axis:
            keyframes = fcurve.keyframe_points  

    # take values from keyframes so we can process them
    data_array = []
    for frame in keyframes:
        data_array.append([frame.co[0], frame.co[1]])
    data_array = np.array(data_array)

    # calculate simple moving average
    weights = np.repeat(1.0, kernel_size)/kernel_size
    sma = np.convolve(data_array[:, 1], weights, 'valid')

    # replace old values with new ones
    start = int(kernel_size/2)
    print(start)
    for idx, keyframe in enumerate(keyframes[start:-start]):
        keyframes.insert(keyframe.co[0], sma[idx], options={'FAST'}, keyframe_type='KEYFRAME')


class My_settings(bpy.types.PropertyGroup):
    view_result : bpy.props.BoolProperty(name='view result', default=True)
    loop_video : bpy.props.BoolProperty(name='loop video', default=True)
    fps: bpy.props.FloatProperty(name="FPS", default=24.0, min=0, max=60)
    scale: bpy.props.FloatProperty(name="video scale", default=.4, min=0, max=1)
    eye_mag: bpy.props.FloatProperty(name="eye mag", default=30, min=1)
    smoothing: bpy.props.IntProperty(name="smoothing", default=9, min=0, max=30) 
    video_path: bpy.props.StringProperty(name="video path", default='') 


class LOAD_OT_load_data(Operator, ImportHelper): 
    bl_idname = "load.load_data" 
    bl_label = "Open the file browser" 
    directory : StringProperty(subtype='DIR_PATH')
    files : CollectionProperty(type=OperatorFileListElement)

    
    def create_face_coor(self ,cap ,path_frames ,frame_rate=24 ,scale=.75 ,view=False):
        '''
        a function that gives you the option of taking a video and exporting
        it as an image sequence then returns mediapipe facemesh as a numpy array.
          
        or you can just view the face mesh since the option above is slow.
        '''
        print(frame_rate)
        cap = cv2.VideoCapture(cap)
        mp_drawing = mp.solutions.drawing_utils
        mp_face_mesh = mp.solutions.face_mesh
        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=3)
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.8)
        coor_list = []
        
        count = 0
        frame = 0
        count_draw = 0
        count_more = 0

        while cap.isOpened():
            if not view:
                cap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))  
            success, img = cap.read()
            
            if not success:
                break

            # reads image            
            else:
                img = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale))) # reduced size
                img.flags.writeable = False
                results = face_mesh.process(img)
                img.flags.writeable = True
                vert_list = []
                
                # saves images 
                if not view: 
                    print('frame', count_more)
                    count_more += 1
                    #path = path_frames + '//' + str(frame) + '.png'
                    #cv2.imwrite(path, img)
                    count += (1/frame_rate)
                    frame += 1

                # collects and draws face mask
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        for vert in face_landmarks.landmark:
                            vert_list.append([vert.x, vert.y, vert.z])
                        coor_list.append(vert_list)
                        
                        # displays image
                        if view:
                            mp_drawing.draw_landmarks(
                                image=img,
                                landmark_list=face_landmarks,
                                connections=mp_face_mesh.FACE_CONNECTIONS,
                                landmark_drawing_spec=drawing_spec,
                                connection_drawing_spec=drawing_spec)
                               
                            cv2.imshow('MediaPipe FaceMesh', img)
                            cv2.waitKey(1)
                            cv2.imwrite(f'E://facemocap//addon//view_markers//{count_draw}.png', img)
                            count_draw += 1

            
        if view:
            cv2.destroyAllWindows()
        return coor_list
    
    def execute(self, context): 
        if not bpy.data.is_saved:
            print('save the file to a project directory first')

        else:
            # lets user find video path and returns it as string
            base = Path(self.directory)                
            for f in self.files:
                video_path = base / f.name
                print(video_path)
            
            video_path = str(video_path)
            my_tool = context.scene.my_tool
            my_tool.video_path = video_path


            # need to change this to: if not video_path.split('.')[0] in list_with_valid_video_formats: pick correct format
            if not video_path.split('.')[1] == 'mp4':
                print('select a mp4 file')

            else:
                # finds blend file directory, creates a folder in it and returns path
                dir = bpy.path.abspath("//") # should prob just put this in function, will fix later.
                path_frames = create_folder(dir, 'frames')

                # create marker data
                my_tool = context.scene.my_tool
                faces_coor = self.create_face_coor(video_path, path_frames, frame_rate=my_tool.fps, scale=my_tool.scale, view=my_tool.view_result)

                # saving numpy for later use.
                np.save(dir + 'faces_array' ,faces_coor)

        return {'FINISHED'}


class TRACK_OT_track_head(Operator):
    bl_idname = "track.head" 
    bl_label = "tracks head rotation" 
    bl_options = {'UNDO'}

    def execute(self, context):
        print('new run')
        # load data
        dir = bpy.path.abspath("//")
        faces = np.load(dir + 'faces_array.npy')
        rotation = []
        # centre face
        faces = adjust_point_cloud_loc(faces, (197)) # 197

        for face in faces:
            # define point to track
            verts = np.array([face[6] ,face[133]])
            target = np.array([1,0])

            # facing z
            verts_transformed, rotation_z = rotate_along_axis(verts, 2, 0, target)
            verts[:, 0] = verts_transformed[:, 0]
            verts[:, 1] = verts_transformed[:, 1]

            # facing y
            verts_transformed, rotation_y = rotate_along_axis(verts, 1, 0, target)
            verts[:, 0] = verts_transformed[:, 0]
            verts[:, 2] = verts_transformed[:, 1]

            # facing x
            verts_transformed, rotation_x = rotate_along_axis(verts, 0, 1, target)
            verts[:, 1] = verts_transformed[:, 0]
            verts[:, 2] = verts_transformed[:, 1]

            rot_left = np.array([rotation_x, rotation_y, rotation_z])

            # again to average
            verts = np.array([face[6] ,face[362]])
            target = np.array([1,0])

            # facing z
            verts_transformed, rotation_z = rotate_along_axis(verts, 2, 0, target)
            verts[:, 0] = verts_transformed[:, 0]
            verts[:, 1] = verts_transformed[:, 1]

            # facing y
            verts_transformed, rotation_y = rotate_along_axis(verts, 1, 0, target)
            verts[:, 0] = verts_transformed[:, 0]
            verts[:, 2] = verts_transformed[:, 1]

            # facing x
            verts_transformed, rotation_x = rotate_along_axis(verts, 0, 1, target)
            verts[:, 1] = verts_transformed[:, 0]
            verts[:, 2] = verts_transformed[:, 1]

            rot_right = np.array([rotation_x, rotation_y, rotation_z])
            rot = (rot_right + rot_left)/2
            
            rotation.append([rot[0], rot[1], rot[2]])
        
        rotation = np.array(rotation)
        print(rotation)
        rotation = rotation - rotation[0]
        print(np.degrees(rotation))
        
        bpy.ops.mesh.primitive_cube_add()   
        bpy.context.active_object.name = 'head_rotation'
        obj = bpy.context.active_object
        for idx, rot in enumerate(rotation):
            obj.rotation_euler[0] = rot[0]
            obj.rotation_euler[1] = rot[1]
            obj.rotation_euler[2] = rot[2]
            obj.keyframe_insert(data_path="rotation_euler", frame=idx)

        my_tool = context.scene.my_tool 
        smooth_curve('head_rotation', 'rotation_euler', 0, my_tool.smoothing)
        smooth_curve('head_rotation', 'rotation_euler', 1, my_tool.smoothing)
        smooth_curve('head_rotation', 'rotation_euler', 2, my_tool.smoothing)

        return {'FINISHED'} 


class TRACK_OT_track_mouth(Operator):
    bl_idname = "track.mouth" 
    bl_label = "tracks mouth rotation" 
    bl_options = {'UNDO'}

    def execute(self, context):
        '''
        lip points:
        -eyes

        *         *
        *   *  362*   *263
        *         *

        -mouth
            *13
        78*      *308
            *14
        '''

        #loading data
        dir = bpy.path.abspath("//")
        faces = np.load(dir + 'faces_array.npy')  
        mouth_idx = [13 ,14, 78, 308, 362, 263]
        mouth_width = [78, 308] 
        mouth_height = [13, 14]
        eye_width = [362, 263]

        # get mags
        mags_width = get_mag(faces[:,mouth_width[0]], faces[:,mouth_width[1]], True)
        mags_height = get_mag(faces[:, mouth_height[0]], faces[:, mouth_height[1]], True)
        mags_scale = get_mag(faces[:, eye_width[0]], faces[:, eye_width[1]], True)

        # get scale
        min = np.min(mags_scale)
        scale = mags_scale / min

        # get normalised distance
        distance_w = norm_arr(mags_width, scale)
        distance_h = norm_arr(mags_height, scale)

        # create cube that'll be used as a driver
        create_empty('mouth_width', distance_w)
        create_empty('mouth_height', distance_h)

        # smoothing
        my_tool = context.scene.my_tool 
        smooth_curve('mouth_width', 'location', 0, my_tool.smoothing)
        smooth_curve('mouth_height', 'location', 0, my_tool.smoothing)

        return {'FINISHED'} 


class TRACK_OT_track_blinks(Operator):
    bl_idname = "track.blinks" 
    bl_label = "tracks eyelids" 
    bl_options = {'UNDO'}

    def execute(self, context):
        # load data
        dir = bpy.path.abspath("//")
        faces = np.load(dir + 'faces_array.npy')

        # top dowm
        left_eye = [386, 374]
        right_eye = [159, 145]
        width_eye = [362, 263]
        faces = np.load('E://youtube//project directory//faces_array.npy')

        # get mags
        mags_left = get_mag(faces[:,left_eye[0]], faces[:,left_eye[1]], True)
        mags_right = get_mag(faces[:,right_eye[0]], faces[:,right_eye[1]], True)
        mags_scale = get_mag(faces[:,width_eye[0]], faces[:,width_eye[1]], True)

        # get scale
        min = np.min(mags_scale)
        scale = mags_scale / min

        # get normalised distance
        distance_l = norm_arr(mags_left, scale)
        distance_r = norm_arr(mags_right, scale)

        # create cube that'll be used as a driver
        create_empty('eye_l', distance_l)
        create_empty('eye_r', distance_r)

        # smoothing
        my_tool = context.scene.my_tool 
        smooth_curve('eye_l', 'location', 0, my_tool.smoothing)
        smooth_curve('eye_r', 'location', 0, my_tool.smoothing)
        return {'FINISHED'}


class TRACK_OT_track_eyes(Operator):
    ''' '''
    bl_idname = "track.eyes" 
    bl_label = "tracks eyes" 
    bl_options = {'UNDO'}

    def execute(self, context):
        '''need to rewrite'''
        dir = bpy.path.abspath("//")
        my_tool = context.scene.my_tool
        loop_video = my_tool.loop_video
        if loop_video:
            # cv2 window and slider settings        
            cv2.namedWindow('image') 
            cv2.createTrackbar('contrast', 'image', 100, 1000, nothing)
            cv2.createTrackbar('brightness', 'image', 0, 100, nothing)
            cv2.createTrackbar('cut off', 'image', 127, 255, nothing)    

        else: 
            params = np.load(dir + '/params.npy')       
            cv2.namedWindow('image') 
            cv2.createTrackbar('contrast', 'image', params[0], 1000, nothing)
            cv2.createTrackbar('brightness', 'image', params[1], 100, nothing)
            cv2.createTrackbar('cut off', 'image', params[2], 255, nothing)

        # load data
        scale = .3
        vid = cv2.VideoCapture(my_tool.video_path)
        iris_locataion = []
        landmarks_list = []

        # mediapipe setting
        mp_drawing = mp.solutions.drawing_utils
        mp_face_mesh = mp.solutions.face_mesh
        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=3)
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.8)

        # idxs for the verts in the face mesh
        right_eye_idx = [33,246,161,160,159,158,157,173,133,155,154,145,144,163,7]
        left_eye_idx = [362,398,384,385,386,387,466,263,249,390,373,374,380,381,382]
        left_brow_idx = [9,336,296,334,293,300,276,283,282,295,285,8]
        right_brow_idx = [8,55,65,52,53,46,70,63,105,66,107,9]

        while(vid.isOpened()):
            # read and resize img
            ret, img = vid.read()  
            
            if ret:
                img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
                img.flags.writeable = False
                results = face_mesh.process(img)
                img.flags.writeable = True
                vert_list = []
                o_img = img
                
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        for vert in face_landmarks.landmark:
                            vert_list.append([int(img.shape[1]*vert.x), int(img.shape[0]*vert.y)])
                    landmarks = np.array(vert_list)
                    landmarks_list.append(landmarks)        
                    
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    alpha = cv2.getTrackbarPos('contrast', 'image')
                    beta = cv2.getTrackbarPos('brightness', 'image')
                    cut = cv2.getTrackbarPos('cut off', 'image')
                    img = cv2.convertScaleAbs(img, alpha=(alpha/100), beta=beta)
                    img_other = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    ret,img = cv2.threshold(img,cut,255,cv2.THRESH_BINARY_INV)
                
                    # create mask to isolate important edges
                    mask = np.zeros(img.shape, dtype=np.uint8)
                    #mask = get_shape(landmarks, right_eye_idx, mask)
                    mask = get_shape(landmarks, left_eye_idx, mask)
                    img = cv2.bitwise_and(img, mask)
                    eye_loc = find_iris(img, o_img)
                    iris_locataion.append(eye_loc)
                    
                    grey_3_channel = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    o_img = np.hstack([o_img, img_other, grey_3_channel])
                else:
                    pass 
                
            else:
                if loop_video:         
                    print('no video')
                    vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
                else: 
                    break
                
                
            cv2.imshow('image', o_img)
            if cv2.waitKey(1) & 0xFF == 27: 
                break
            
        cv2.destroyAllWindows() 
        
        np.save(dir + 'iris_locataion.npy', iris_locataion)
        np.save(dir + 'landmark_list.npy', landmarks_list)
        np.save(dir + 'params.npy', np.array([alpha, beta, cut]))

        if not loop_video:
            def is_postive(number):
                if number >= 0:
                    return True
                else: 
                    return False

            iris_location = np.load(dir + 'iris_locataion.npy')
            landmarks_list = np.load(dir + 'landmark_list.npy')
            eye_marker_idx = (263, 362)  

            point1 = landmarks_list[:, eye_marker_idx[0]]
            point2 = landmarks_list[:, eye_marker_idx[1]]

            x = point2[:, 0] - point1[:, 0]  
            x = (x*.5) + point1[:, 0] 
            y = point2[:, 1] - point1[:, 1]  
            y = (y*.5) + point1[:, 1] 

            new_iris = []
            for i in range(len(iris_location)):
                iris_x = iris_location[i][0] - x[i]
                iris_y = iris_location[i][1] - y[i]
                new_iris.append([iris_x, iris_y])
            
            new_iris = np.array(new_iris)
            new_iris[:, 0] -= new_iris[0][0]
            new_iris[:, 1] -= new_iris[0][1]

            new_iris[:, 1] *= -1


            '''
            bpy.ops.mesh.primitive_cube_add()
            obj = bpy.context.active_object
            obj.name = 'iris_new'  
            for i, d in enumerate(new_iris):
                obj.location[0] = d[0]
                obj.location[1] = d[1]
                obj.keyframe_insert(data_path="location", frame=i)
            '''

            x = new_iris[:, 0]
            y = new_iris[:, 1]
            angles = np.arctan(y/x)


            new_angles = []
            for idx, angle in enumerate(angles):
                x_b = is_postive(x[idx])
                y_b = is_postive(y[idx])

                if x_b == y_b:
                    if x[idx] >= 0:
                        new_angles.append(math.radians(90) - abs(angle))
                    else:
                        new_angles.append(math.radians(270) - abs(angle))
                else:
                    if x[idx] >= 0:
                        new_angles.append(math.radians(90) + abs(angle))
                    else:
                        new_angles.append(math.radians(270) + abs(angle))

            #calulate_mag
            mag = np.sqrt(x**2 + y**2)
            mag_norm = norm_arr(mag, 1)


            x_new = []
            y_new = []
            for idx, angle in enumerate(new_angles):
                x_new.append(math.sin(angle) * (my_tool.eye_mag * mag_norm[idx]))
                y_new.append(math.cos(angle) * (my_tool.eye_mag * mag_norm[idx]))

            x_new = np.array(x_new)
            y_new = np.array(y_new)

            nan_list = np.argwhere(np.isnan(np.array(x_new)))       
            for idx in nan_list:
                x_new[idx] == 0

            nan_list = np.argwhere(np.isnan(np.array(y_new)))             
            for idx in nan_list:
                y_new[idx] == 0

            bpy.ops.mesh.primitive_cube_add()   
            bpy.context.active_object.name = 'eye_pos'
            obj = bpy.context.active_object
            for i, x in enumerate(x_new):
                obj.location[0] = x
                obj.location[1] = y_new[i]
                obj.keyframe_insert(data_path="location", frame=i)

        return {'FINISHED'}



class VIEW3D_PT_value(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'close2mocap'
    bl_label = 'FACE MOCAP'
    
    def draw(self, context):
        self.layout.label(text="face tracking")
        row = self.layout.row()
        my_tool = context.scene.my_tool
        row.prop(my_tool, "fps")
        row.prop(my_tool, "scale")
        row.prop(my_tool, "smoothing")
        row.prop(my_tool, "eye_mag")
        row.prop(my_tool, "view_result")
        row.prop(my_tool, "loop_video")

        self.layout.operator('load.load_data', text='select mp4 file')
        self.layout.operator('track.head', text='track head rotation')
        self.layout.operator('track.blinks', text='track blinks')
        self.layout.operator('track.mouth', text='track mouth')
        self.layout.operator('track.eyes', text='track eyes')

        self.layout.label(text="(var * (max - min)) +  min")
        

        

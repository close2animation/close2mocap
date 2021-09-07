import bpy
from pathlib import Path
from bpy_extras.io_utils import ImportHelper 
from bpy.types import Operator, OperatorFileListElement
from bpy.props import CollectionProperty, StringProperty
import mediapipe as mp
import cv2
import numpy as np
from .rotation import *
import os
from .distance_utils import  *

bpy.types.Scene.target = bpy.props.PointerProperty(type=bpy.types.Object)

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



    data = []
    for obj in bpy.data.collections[name].objects:
        fcurves = get_specific_fcurves(obj, 'location', 'all')
        location = []
        for fcurve in fcurves:
            location.append(fcurve_to_list(fcurve))
        data.append(location)
    return data


class My_settings(bpy.types.PropertyGroup):
    view_result : bpy.props.BoolProperty(name='view result', default=True)
    loop_video : bpy.props.BoolProperty(name='loop video', default=True)
    fps: bpy.props.FloatProperty(name="FPS", default=24.0, min=0, max=60)
    scale: bpy.props.FloatProperty(name="video scale", default=.4, min=0, max=1)
    eye_mag: bpy.props.FloatProperty(name="eye mag", default=30, min=1)
    smoothing: bpy.props.IntProperty(name="smoothing", default=9, min=0, max=30) 
    video_path: bpy.props.StringProperty(name="video path", default='') 
    alpha: bpy.props.IntProperty(name="smoothing", default=100) 
    beta: bpy.props.IntProperty(name="smoothing", default=0) 
    cutoff: bpy.props.IntProperty(name="smoothing", default=127) 
    kernely: bpy.props.IntProperty(name="cutoff", default=1) 
    kernelx: bpy.props.IntProperty(name="cutoff", default=1)


class TRACK_OT_load_data(Operator, ImportHelper):  
    bl_idname = "load.data" 
    bl_label = "load data" 
    bl_options = {'UNDO'}
    directory : StringProperty(subtype='DIR_PATH')
    files : CollectionProperty(type=OperatorFileListElement)


    def load_mp_tools(self):
        self.mp_drawing = mp.solutions.drawing_utils 
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose     
        self.face_mesh = self.mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) 
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


    def create_empty_for_every_landmark(self, name, total_landmarks):
        collection = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(collection)
        bpy.context.view_layer.active_layer_collection = bpy.context.view_layer.layer_collection.children[name]    
               
        for idx in range(total_landmarks):
            bpy.ops.object.empty_add(type='PLAIN_AXES')
            obj = bpy.context.active_object
            obj.name = name + '_' + str(idx) 
                  

    def save_landmarks_xyz_to_empties(self, landmarks, name, frame):
        try:
            for landmark_idx, landmark in enumerate(landmarks.landmark):
                empty_name = name + '_' + str(landmark_idx)   
                obj =  bpy.data.objects[empty_name]
                obj.location[0] = landmark.x
                obj.location[1] = landmark.y
                obj.location[2] = landmark.z
                obj.keyframe_insert(data_path="location", frame=frame)
        except:
            pass
   
   
    def draw_results(self, img, face_results, hand_results, pose_results):

        if face_results.multi_face_landmarks:      
            self.mp_drawing.draw_landmarks(
                img,
                face_results.multi_face_landmarks[0],
                self.mp_face_mesh.FACE_CONNECTIONS)
            
        if hand_results.multi_hand_landmarks:  
            self.mp_drawing.draw_landmarks(
                img,
                hand_results.multi_hand_landmarks[0],
                self.mp_hands.HAND_CONNECTIONS)
            try:
                self.mp_drawing.draw_landmarks(
                    img,
                    hand_results.multi_hand_landmarks[1],
                    self.mp_hands.HAND_CONNECTIONS)  
            except:
                pass 

        if pose_results.pose_landmarks:          
            self.mp_drawing.draw_landmarks(
                img,
                pose_results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS)     
                    
        cv2.imshow('result', img)
        cv2.waitKey(1)  
        
    
    def loop_through(self, operation, landmark_list, name, frame):
        '''
        need to loop through face and hands since more than face/hand can exist at 1 time. 
        also need to check if landsmarks exist.
        '''  
        if not landmark_list:
            print(f'no results in frame{frame} for {name}')
            pass
        else:
            try:
                for idx, landmarks in enumerate(landmark_list):
                    #print(idx)
                    operation(landmarks, name[idx], frame)
            except IndexError:
                pass
            
    
    def execute(self, context):
        # lets user find video path and returns it as string
        base = Path(self.directory)                
        for f in self.files:
            video_path = base / f.name
            print(video_path)
        
        video_path = str(video_path)
        my_tool = context.scene.my_tool
        my_tool.video_path = video_path

        # need to change this to: if not video_path.split('.')[0] in list_with_valid_video_formats: pick correct format
        if not video_path.split('.')[-1] == 'mp4':
            print('select a mp4 file')
        
        else:
            #params
            cap = cv2.VideoCapture(my_tool.video_path)
            scale = my_tool.scale
            frame = 0
            view_results = my_tool.view_result
            frame_rate= my_tool.fps
            count = 0
            lock_fps = my_tool.view_result
            hand_side = ['hand_l', 'hand_r']
            
            self.load_mp_tools()
            self.create_empty_for_every_landmark('face', 468)
            self.create_empty_for_every_landmark(hand_side[0], 21)
            self.create_empty_for_every_landmark(hand_side[1], 21)
            self.create_empty_for_every_landmark('pose', 33)
                        
            while cap.isOpened():           
                if not lock_fps:
                    cap.set(cv2.CAP_PROP_POS_MSEC,(count*1000)) # doing this takes twice as long.....
                    count += (1/frame_rate)      
                success, img = cap.read()
                if not success:
                    break
                    
                # process img
                img = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))
                img = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)
                img.flags.writeable = False
                face_results = self.face_mesh.process(img)
                hand_results = self.hands.process(img)
                pose_results = self.pose.process(img)
                
                if view_results:
                    img.flags.writeable = True
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    self.draw_results(img, face_results, hand_results, pose_results)
                
                self.loop_through(self.save_landmarks_xyz_to_empties, face_results.multi_face_landmarks, ['face'], frame)
                self.loop_through(self.save_landmarks_xyz_to_empties, hand_results.multi_hand_landmarks, hand_side, frame) 
                if pose_results.pose_landmarks:
                    self.save_landmarks_xyz_to_empties(pose_results.pose_landmarks, 'pose', frame) 
                frame += 1     
            cv2.destroyAllWindows()
        return {'FINISHED'}


class TRACK_OT_track_head(Operator):
    bl_idname = "track.head" 
    bl_label = "tracks head rotation" 
    bl_options = {'UNDO'}

    def execute(self, context):
        data_face = collect_data_in_collection('face')
        # set data shape to frames/obj/xyz
        data_face = np.transpose(np.array(data_face), (2, 0, 1))
        
        # centre face
        faces = adjust_point_cloud_loc(data_face, (197)) # 197
        rotation = []

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
        # load data
        data_face = collect_data_in_collection('face')
        # set data shape to frames/obj/xyz
        faces = np.transpose(np.array(data_face), (2, 0, 1))

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
        data_face = collect_data_in_collection('face')
        # set data shape to frames/obj/xyz
        faces = np.transpose(np.array(data_face), (2, 0, 1))

        # top dowm
        left_eye = [386, 374]
        right_eye = [159, 145]
        width_eye = [362, 263]

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
        my_tool = context.scene.my_tool 
        loop_video = my_tool.loop_video

        # cv2 window and slider settings        
        cv2.namedWindow('image') 
        cv2.createTrackbar('contrast', 'image', my_tool.alpha, 1000, nothing)
        cv2.createTrackbar('brightness', 'image', my_tool.beta, 100, nothing)
        cv2.createTrackbar('cut off', 'image', my_tool.cutoff, 255, nothing)   
        cv2.createTrackbar('kernelx', 'image', my_tool.kernelx, 15, nothing) 
        cv2.createTrackbar('kernely', 'image', my_tool.kernely, 15, nothing)  

        # load data
        faces = collect_data_in_collection('face') 
        faces = np.transpose(np.array(faces), (2, 0, 1))  # set data shape to frames/obj/xyz
        print(faces.shape)
        scale = my_tool.scale
        vid = cv2.VideoCapture(my_tool.video_path)
        iris_location = []
        frame = 0
       
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
                img = cv2.flip(img, 1)
                o_img = img
                     
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                alpha = cv2.getTrackbarPos('contrast', 'image')
                beta = cv2.getTrackbarPos('brightness', 'image')
                cut = cv2.getTrackbarPos('cut off', 'image')
                kx = cv2.getTrackbarPos('kernelx', 'image')
                ky = cv2.getTrackbarPos('kernely', 'image')
                img = cv2.convertScaleAbs(img, alpha=(alpha/100), beta=beta)
                img_other = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                ret,img = cv2.threshold(img,cut,255,cv2.THRESH_BINARY_INV)
            
                # create mask to isolate important edges
                mask = np.zeros(img.shape, dtype=np.uint8)
                #mask = get_shape(landmarks, right_eye_idx, mask)
                mask = get_shape(faces[frame], left_eye_idx, mask, kx, ky)
                img = cv2.bitwise_and(img, mask)
                eye_loc = find_iris(img, o_img)
                iris_location.append(eye_loc)
                
                grey_3_channel = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                o_img = np.hstack([o_img, img_other, grey_3_channel])
                frame += 1

                
            else:
                if loop_video:         
                    print('no video')
                    vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame = 0
                else: 
                    break
                
                
            cv2.imshow('image', o_img)
            if cv2.waitKey(1) & 0xFF == 27: 
                break

        cv2.destroyAllWindows() 
        my_tool.alpha = alpha
        my_tool.beta = beta
        my_tool.cutoff = cut
        my_tool.kernely = ky
        my_tool.kernelx = kx

        if not loop_video:    
            faces[:, :, 0] *= mask.shape[1]
            faces[:, :, 1] *= mask.shape[0]            
            
            # eye_marker_idx = (263, 362) 
            point1 = faces[:, 263, :2]
            point2 = faces[:, 362, :2]

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

            x = new_iris[:, 0]
            y = new_iris[:, 1]
            angles = np.arctan2(y, x)

            #calulate_mag
            mag = np.sqrt(x**2 + y**2)
            mag_norm = norm_arr(mag, 1)

            x_new = []
            y_new = []
            for idx, angle in enumerate(angles):
                x_new.append(math.sin(angle) * (my_tool.eye_mag * mag_norm[idx]))
                y_new.append(math.cos(angle) * (my_tool.eye_mag * mag_norm[idx]))

            x_new = np.array(x_new)
            y_new = np.array(y_new)

            bpy.ops.mesh.primitive_cube_add()   
            bpy.context.active_object.name = 'eye_pos'
            obj = bpy.context.active_object
            for i, x in enumerate(x_new):
                obj.location[0] = y_new[i]
                obj.location[1] = x
                obj.keyframe_insert(data_path="location", frame=i)
                
        return {'FINISHED'}


class TRACK_OT_track_fingers(Operator):
    ''' '''
    bl_idname = "track.fingers" 
    bl_label = "tracks fingers" 
    bl_options = {'UNDO'}
    left : bpy.props.BoolProperty()

    def execute(self, context):
        print('new run')
        print(self.left)
        
        if self.left == True:
            data_hand = collect_data_in_collection('hand_l')
        else:
            data_hand = collect_data_in_collection('hand_r')   
            
        # set data shape to frames/obj/xyz
        data_hand = np.transpose(np.array(data_hand), (2, 0, 1))
        angles1 = select_finger(data_hand, 5, 9)
        angles2 = select_finger(data_hand, 9, 13)
        angles3 = select_finger(data_hand, 13, 17)
        angles4 = select_finger(data_hand, 17, 21) 
        angles5 = select_finger(data_hand, 1, 5) 

        angles1 = np.array(angles1)
        angles2 = np.array(angles2)
        angles3 = np.array(angles3)
        angles4 = np.array(angles4)  
        angles5 = np.array(angles5) 

        offsets = [math.radians(-15), math.radians(-15), math.radians(-15)]
        offsets2 = [math.radians(-25), math.radians(-15), math.radians(-15)]
        obj = bpy.data.objects['rig']
        finger_labels = ['thumb', 'f_index', 'f_middle', 'f_ring', 'f_pinky']

        add_animation_to_finger(angles1, finger_labels[1], offsets)
        add_animation_to_finger(angles2, finger_labels[2], offsets)
        add_animation_to_finger(angles3, finger_labels[3], offsets)
        add_animation_to_finger(angles4, finger_labels[4], offsets)
        add_animation_to_finger(angles5, finger_labels[0], offsets2)
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

        self.layout.operator('load.data', text='select mp4 file')
        self.layout.operator('track.head', text='track head rotation')
        self.layout.operator('track.blinks', text='track blinks')
        self.layout.operator('track.mouth', text='track mouth')
        self.layout.operator('track.eyes', text='track eyes')
        op = self.layout.operator('track.fingers', text='fingers left')
        op.left = True
        op = self.layout.operator('track.fingers', text='fingers right')
        op.left = False
        self.layout.prop_search(context.scene, "target", context.scene, "objects", text="hand")

        self.layout.label(text="(var * (max - min)) +  min")
        

        

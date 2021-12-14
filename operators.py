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
from .eye_utils import  *
from .rotation_quat import *

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
    lock_fps : bpy.props.BoolProperty(name='lock fps', default=False)
    t_hand : bpy.props.BoolProperty(name='track hand', default=True)
    t_head : bpy.props.BoolProperty(name='track head', default=True)
    t_pose : bpy.props.BoolProperty(name='track pose', default=True)
    invert : bpy.props.BoolProperty(name='invert y and z', default=True)
    inverse : bpy.props.BoolProperty(name='invert direction', default=True)


class TRACK_OT_load_data(Operator, ImportHelper):  
    bl_idname = "load.data" 
    bl_label = "load data" 
    bl_options = {'UNDO'}
    directory : StringProperty(subtype='DIR_PATH')
    files : CollectionProperty(type=OperatorFileListElement)


    def load_mp_tools(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose     
        self.face_mesh = self.mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5, refine_landmarks=True) 
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

    
    def create_camera(self, video_path):    
        bpy.ops.object.camera_add()
        bpy.data.scenes['Scene'].render.resolution_x = 1000
        bpy.data.scenes['Scene'].render.resolution_y = 1000
        camera = bpy.context.active_object
        camera.name = 'visualise points'
        camera.location[0] = .5
        camera.location[1] = .5
        camera.location[2] = -1
        camera.rotation_euler[0] = math.radians(180)
        camera.rotation_euler[1] = 0
        camera.rotation_euler[2] = 0
        camera.data.type = 'ORTHO'
        camera.data.ortho_scale = 1.0
        camera.data.show_background_images = True

        video = bpy.data.movieclips.load(video_path)
        bg = camera.data.background_images.new()
        bg.source = 'MOVIE_CLIP'
        bg.clip = video
        bg.rotation = math.radians(180)
        bg.use_flip_y = True
            
    
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
            lock_fps = my_tool.lock_fps
            hand_side = ['hand_l', 'hand_r']
            t_hand = my_tool.t_hand
            t_head = my_tool.t_head
            t_pose = my_tool.t_pose
            self.load_mp_tools()
            self.create_camera(video_path)

            if t_head:
                self.create_empty_for_every_landmark('face', 478)
            if t_hand:
                self.create_empty_for_every_landmark(hand_side[0], 21)
                self.create_empty_for_every_landmark(hand_side[1], 21)
            if t_pose:
                self.create_empty_for_every_landmark('pose', 33)
                        
            while cap.isOpened():           
                if lock_fps:
                    cap.set(cv2.CAP_PROP_POS_MSEC,(count*1000)) # doing this takes twice as long.....
                    count += (1/frame_rate)      
                success, img = cap.read()
                if not success:
                    break
                    
                # process img
                img = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))
                #img = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img.flags.writeable = False

                face_results = None
                hand_results = None
                pose_results = None
                if t_head:
                    face_results = self.face_mesh.process(img)
                    self.loop_through(self.save_landmarks_xyz_to_empties, face_results.multi_face_landmarks, ['face'], frame)
                if t_hand:
                    hand_results = self.hands.process(img)
                    self.loop_through(self.save_landmarks_xyz_to_empties, hand_results.multi_hand_landmarks, hand_side, frame) 
                if t_pose:
                    pose_results = self.pose.process(img)
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

        #my_tool = context.scene.my_tool 
        #smooth_curve('head_rotation', 'rotation_euler', 0, my_tool.smoothing)
        #smooth_curve('head_rotation', 'rotation_euler', 1, my_tool.smoothing)
        #smooth_curve('head_rotation', 'rotation_euler', 2, my_tool.smoothing)

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
        #my_tool = context.scene.my_tool 
        #smooth_curve('mouth_width', 'location', 0, my_tool.smoothing)
        #smooth_curve('mouth_height', 'location', 0, my_tool.smoothing)

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
        #my_tool = context.scene.my_tool 
        #smooth_curve('eye_l', 'location', 0, my_tool.smoothing)
        #smooth_curve('eye_r', 'location', 0, my_tool.smoothing)
        return {'FINISHED'}


class TRACK_OT_track_eyes(Operator):
    ''' '''
    bl_idname = "track.eyes" 
    bl_label = "tracks eyes" 
    bl_options = {'UNDO'}

    def execute(self, context):
        '''need to rewrite'''

        # load data
        faces = collect_data_in_collection('face') 
        faces = np.transpose(np.array(faces), (2, 0, 1))  # set data shape to frames/obj/xyz
        print(faces.shape)
        
        # v idx is 386 and 253
        # eye_marker_idx = (263, 362) 
        point1 = faces[:, 263, :2]
        point2 = faces[:, 362, :2]

        # finding midpoint between 263 and 362
        x = point2[:, 0] - point1[:, 0]  
        x = (x*.5) + point1[:, 0] 
        y = point2[:, 1] - point1[:, 1]  
        y = (y*.5) + point1[:, 1] 

        # 468 and 473 are iris points. 473 being the left side
        iris_location = faces[:, 473, :2]

        # removing head motion from iris. sort of
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
            x_new.append(math.sin(angle) * (mag_norm[idx]))
            y_new.append(math.cos(angle) * (mag_norm[idx]))

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


class TRACK_OT_track_to_rig(Operator):
    ''' '''
    bl_idname = "track.rig" 
    bl_label = "tracks to rig" 
    bl_options = {'UNDO'}

    def execute(self, context):
        my_tool = context.scene.my_tool
        # lazy method for now
        # head --------------------------------------------------------------------------------------------

        # load data
        data_face = collect_data_in_collection('face')
        data_face = data_face[:468]
        # set data shape to frames/obj/xyz
        faces = np.transpose(np.array(data_face), (2, 0, 1))
        faces = move_to_world_origin(faces, 168)

        p1 = faces[:, 9]
        p2 = faces[:, 244]

        quats = generate_quaternion_batch(p1, p2)
        offsets = offset_quat(quats, quats[0])


        # test with many
        verts = np.transpose(faces, (1, 0, 2))
        rotations = np.zeros((verts.shape[1], 4))
        for i, vert in enumerate(verts[:3]):
            
            print('p1 and verts')
            print(p1.shape)
            print(vert.shape)
            quats = generate_quaternion_batch(p1, vert)
            offsets = offset_quat(quats, quats[0])
            rotations += np.array(offsets)

        print(offsets)
        rotations = rotations / (i+1)
        rotations = normalise_vector_batch(rotations)
        obj = bpy.context.scene.target
        apply_rotation_to_rig_quat(rotations, obj, False)


        #body ---------------------------------------------------------------------
        data_face = collect_data_in_collection('pose')
        # set data shape to frames/obj/xyz
        data_pose = np.transpose(np.array(data_face), (2, 0, 1))
        data_pose = data_pose

        # shoulder points
        left = data_pose[:, 11]
        right = data_pose[:, 12]
        middle_top = get_middle_point_2d(left, right)

        # making target sideways since bone we using to control is perpendicular to line we're rotating
        # e.g    shoulder line --->    *------------*
        #                                    |
        #                                    |
        #                   bone line --->   |
        target = np.array([1, 0, 0])
        quats_top = slope_angle_2d(right[:, :2], left[:, :2], target)

        # middle point between waist 
        left = data_pose[:, 24]
        right = data_pose[:, 23]
        middle_bot = get_middle_point_2d(left, right)

        # bot coors are actually higher since the camera looks upwards not downwards
        # so we make target negative
        target = np.array([0, -1, 0])
        quats_bot = slope_angle_2d(middle_bot, middle_top, target)

        if my_tool.inverse:
            d1 = True
            d2 = False
        else:
            d1 = False
            d2 = True
        
        if my_tool.invert:
            quats_bot[:, [3, 2]] = quats_bot[:, [2, 3]]
            quats_top[:, [3, 2]] = quats_top[:, [2, 3]]

        apply_quaternion_to_bone(quats_bot, obj, 'torso', d1)
        apply_quaternion_to_bone(quats_bot, obj, 'chest', d2)
        add_rotation_q_to_bone(quats_top, obj, 'chest', d1)

        
        # shoulder points again
        left = data_pose[:, 11]
        right = data_pose[:, 12]

        # drop y axis
        left = np.delete(left, 1, axis=1)
        right = np.delete(right, 1, axis=1)
        target = np.array([1, 0, 0])
        quats = slope_angle_2d(right, left, target)

        if not my_tool.invert: 
            quats[:, [3,2]] = quats[:,[2,3]]

        add_rotation_q_to_bone(quats, obj, 'torso', d2)

        return {'FINISHED'}
        


class VIEW3D_PT_load_data(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'close2mocap'
    bl_label = 'close2mocap'
    
    def draw(self, context):     
        self.layout.label(text="processing options")
        my_tool = context.scene.my_tool
        
        row = self.layout.row()
        row.prop(my_tool, "fps")
        row.prop(my_tool, "lock_fps")
        row.prop(my_tool, "scale")
        
        row = self.layout.row()
        row.prop(my_tool, "t_head")
        row.prop(my_tool, "t_hand")
        row.prop(my_tool, "t_pose")
        self.layout.operator('load.data', text='select mp4 file')


class VIEW3D_PT_track(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'close2mocap'
    bl_label = 'track animation'
    
    def draw(self, context): 
        my_tool = context.scene.my_tool    
        self.layout.operator('track.head', text='track head rotation')
        self.layout.operator('track.blinks', text='track blinks')
        self.layout.operator('track.mouth', text='track mouth')
        self.layout.operator('track.eyes', text='track eyes')
        
        op = self.layout.operator('track.fingers', text='fingers right')
        op.left = True
        op = self.layout.operator('track.fingers', text='fingers left')
        op.left = False

        
        self.layout.prop_search(context.scene, "target", context.scene, "objects", text="rig")
        self.layout.operator('track.rig', text='track to rig')
        row = self.layout.row()
        row.prop(my_tool, "invert")
        row.prop(my_tool, "inverse")

        self.layout.label(text=" normalise range formula: (var * (max - min)) +  min")
        

        

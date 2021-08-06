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


class My_settings(bpy.types.PropertyGroup):
    view_result : bpy.props.BoolProperty(name='view result', default=True)
    fps: bpy.props.FloatProperty(name="FPS", default=24.0)
    scale: bpy.props.FloatProperty(name="video scale", default=.75)


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
                    print('frame', count)
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
            verts = np.array([face[168] ,face[33]])
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
            
            rotation.append([rotation_x, rotation_y, rotation_z])
        
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
        row.prop(my_tool, "view_result")

        self.layout.operator('load.load_data', text='select mp4 file')
        self.layout.operator('track.head', text='track head rotation')

        
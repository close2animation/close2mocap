'''
Copyright 2021 Manuel Acevedo

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.	
'''



import bpy
import sys
import subprocess
import ensurepip
import os
import imp

bl_info = {
    "name" : "close2mocap",
    "author" : "manuel acevedo",
    "description" : "an addon that uses mediapipe to track head rotation. more features coming soon...ish",
    "blender" : (2, 80, 0),
    "version" : (0, 0, 1),
    "location" : "",
    "warning" : "THIS ADDON WILL INSTALL 3RD PARTY LIBRARIES",
    "category" : "motion capture"
}

def install_libs(): 
    ensurepip.bootstrap()
    os.environ.pop("PIP_REQ_TRACKER", None)
	
    # forcing the path cause i can
    version = bpy.app.version_string
    version = int(version[2])

    path = sys.executable
    path = path.split('bin')[0]
    path = path + 'lib\site-packages'
    python_path = sys.executable

    print('this is the path look here', path)

    subprocess.check_output([python_path, '-m', 'pip', 'install', 'opencv-python', '-t', path])
    subprocess.check_output([python_path, '-m', 'pip', 'install', 'mediapipe', '-t', path])
    subprocess.check_output([python_path, '-m', 'pip', 'install', '--ignore-installed', 'six', '-t', path])
    subprocess.check_output([python_path, '-m', 'pip', 'install', '--ignore-installed', 'attrs', '-t', path])
    subprocess.check_output([python_path, '-m', 'pip', 'install', '--ignore-installed', 'matplotlib', '-t', path])

try:
    imp.find_module('mediapipe')

except ImportError:
    install_libs()
	
	
from .operators import  *


classes = (
    My_settings,
    TRACK_OT_load_data,
    VIEW3D_PT_load_data,
    VIEW3D_PT_track_to_cube,
    VIEW3D_PT_track_to_rig,
    TRACK_OT_track_head,
    TRACK_OT_track_mouth,
    TRACK_OT_track_blinks,
    TRACK_OT_track_eyes,
    TRACK_OT_track_fingers,
    TRACK_OT_track_to_body,
    TRACK_OT_track_to_head,
    TRACK_OT_create_animation
) 


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.my_tool = bpy.props.PointerProperty(type=My_settings)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.my_tool


if __name__ == "__main__":
    register()

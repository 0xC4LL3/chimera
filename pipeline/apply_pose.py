# Key a fitted pose (from fit.py) onto the rig and render it from the reference camera.
# Usage: blender -b --python pipeline/apply_pose.py -- <in.glb> <run_dir> <pose.npz> <out_dir>
import bpy, sys, os, json
import numpy as np
from mathutils import Matrix

sys.path.append(os.path.dirname(__file__))
import rig

glb, run, pose, out = rig.args()[:4]
cam_info = json.load(open(os.path.join(run, "camera.json")))
os.makedirs(out, exist_ok=True)
data = np.load(pose)

arm, _ = rig.load(glb)
arm.animation_data.action = bpy.data.actions.new("Fit")
for i, frame in enumerate(data["basis"]):
    for name, m in zip(data["names"], frame):
        pb = arm.pose.bones[str(name)]
        pb.matrix_basis = Matrix(m.tolist())
        pb.keyframe_insert("location", frame=i)
        pb.keyframe_insert("rotation_quaternion", frame=i)
rig.render_from_camera(cam_info, out, len(data["basis"]))

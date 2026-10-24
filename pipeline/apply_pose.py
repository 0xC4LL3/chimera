# Key a fitted pose (from fit.py) onto the rig and render it from the reference camera, and
# again from a camera turned 60 degrees around the creature (<out_dir>_turned), which shows the
# motion towards and away from the reference camera that its own view cannot.
# Usage: python pipeline/apply_pose.py <in.glb> <run_dir> <pose.npz> <out_dir>
import bpy, sys, os, json, math
import numpy as np
from mathutils import Matrix, Vector

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

center = sum((b.head_local for b in arm.data.bones), Vector()) / len(arm.data.bones)
cam = bpy.context.scene.camera
cam.matrix_world = Matrix(cam_info["cam_to_world"])
cam.matrix_world = Matrix.Translation(center) @ Matrix.Rotation(math.radians(60), 4, "Z") @ Matrix.Translation(-center) @ cam.matrix_world
bpy.context.scene.render.filepath = os.path.join(out + "_turned", "")
bpy.context.scene.render.resolution_percentage = 50  # only for looking at
bpy.ops.render.render(animation=True)

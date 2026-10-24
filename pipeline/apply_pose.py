# Key a fitted pose (from fit.py) onto the rig and render it from the reference camera, and
# again from a camera turned 60 degrees around the creature (<out_dir>_turned), which shows the
# motion towards and away from the reference camera that its own view cannot. A split-screen run
# (render_ref.py --split) renders every view and puts them side by side like the clip.
# Usage: python pipeline/apply_pose.py <in.glb> <run_dir> <pose.npz> <out_dir>
import bpy, sys, os, json, math, glob, shutil
import cv2, numpy as np
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
views = cam_info.get("views", [])
if not views:
    rig.render_from_camera(cam_info, out, len(data["basis"]))
else:
    x0, x1 = views[0]["box"]
    rig.render_from_camera({**cam_info, "width": x1 - x0}, out + "_view0", len(data["basis"]))
    for i, v in enumerate(views[1:], 1):
        bpy.context.scene.camera.matrix_world = Matrix(v["cam_to_world"])
        bpy.context.scene.render.filepath = os.path.join(out + f"_view{i}", "")
        bpy.ops.render.render(animation=True)
    dirs = [out + f"_view{i}" for i in range(len(views))]
    for frame in sorted(os.listdir(dirs[0])):
        if frame.endswith(".png"):
            cv2.imwrite(os.path.join(out, frame), np.hstack([cv2.imread(os.path.join(d, frame), cv2.IMREAD_UNCHANGED) for d in dirs]))
    for d in dirs:
        shutil.rmtree(d)
    bpy.context.scene.render.resolution_x = cam_info["width"]

center = sum((b.head_local for b in arm.data.bones), Vector()) / len(arm.data.bones)
cam = bpy.context.scene.camera
cam.matrix_world = Matrix(cam_info["cam_to_world"])
cam.matrix_world = Matrix.Translation(center) @ Matrix.Rotation(math.radians(60), 4, "Z") @ Matrix.Translation(-center) @ cam.matrix_world
bpy.context.scene.render.filepath = os.path.join(out + "_turned", "")
bpy.context.scene.render.resolution_percentage = 50  # only for looking at
bpy.ops.render.render(animation=True)

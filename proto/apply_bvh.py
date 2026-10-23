# Apply a MoCapAnything BVH to the original rig and render it from the reference camera.
# Usage: blender -b --python proto/apply_bvh.py -- <in.glb> <run_dir> <pred.bvh> <out_dir> [n_frames]
# The BVH skeleton is the same rig, rotated to face +Z and without root translation. We carry
# over each bone's world-space rotation change from its rest pose, then align the whole pose
# to the known reference pose on frame 0 (the video's first frame is our render).
# Writes <out_dir>/####.png (RGBA) and <out_dir>/<name>.blend for inspection.
import bpy, sys, os, json
import numpy as np
from mathutils import Matrix, Vector

sys.path.append(os.path.dirname(__file__))
import rig

glb, run, bvh, out = sys.argv[sys.argv.index("--") + 1:][:4]
extra = sys.argv[sys.argv.index("--") + 5:]
cam_info = json.load(open(os.path.join(run, "camera.json")))
os.makedirs(out, exist_ok=True)


def kabsch(src, dst):
    # Rotation R (and scale s) minimising |s R (src - mean) - (dst - mean)|.
    a, b = src - src.mean(0), dst - dst.mean(0)
    u, sv, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1, 1, d]) @ u.T
    return r, (sv * [1, 1, d]).sum() / (a ** 2).sum()


arm, _ = rig.load(glb)
scene = bpy.context.scene
bones = list(arm.pose.bones)
names = [pb.name for pb in bones]
rest = {b.name: b.matrix_local.copy() for b in arm.data.bones}

# Reference pose (frame 0 of the video) in world space.
arm.animation_data.action = bpy.data.actions[cam_info["action"]]
scene.frame_set(cam_info["frame"])
ref_pos = np.array([arm.pose.bones[n].head for n in names])
arm.animation_data.action = None
for pb in bones:
    pb.matrix_basis.identity()

bpy.ops.import_anim.bvh(filepath=bvh, rotate_mode="NATIVE", update_scene_fps=False, update_scene_duration=False)
pred = next(o for o in scene.objects if o.type == "ARMATURE" and o != arm)
pa = pred.animation_data.action
p0, p1 = map(int, pa.frame_range)
n_out = int(extra[0]) if extra else p1 - p0 + 1

# Rigid map from the BVH rest skeleton to ours (rest poses are the same shape).
pred_rest = {b.name: pred.matrix_world @ b.matrix_local for b in pred.data.bones}
A, _ = kabsch(np.array([pred_rest[n].to_translation() for n in names]), np.array([rest[n].to_translation() for n in names]))
A = Matrix(A.tolist())


def pose_at(f):
    # World rotation per bone carried over from the BVH, positions from FK on our offsets.
    scene.frame_set(f)
    world = {}
    for pb in bones:
        q = pred.matrix_world @ pred.pose.bones[pb.name].matrix
        delta = q.to_3x3() @ pred_rest[pb.name].to_3x3().inverted()
        rot = A @ delta @ A.inverted() @ rest[pb.name].to_3x3()
        if pb.parent:
            head = (world[pb.parent.name] @ rest[pb.parent.name].inverted() @ rest[pb.name]).to_translation()
        else:
            head = rest[pb.name].to_translation()
        world[pb.name] = Matrix.Translation(head) @ rot.to_4x4()
    return world


# Global alignment on frame 0, then key every frame on our rig.
w0 = pose_at(p0)
pos0 = np.array([w0[n].to_translation() for n in names])
C, _ = kabsch(pos0, ref_pos)
C = Matrix(C.tolist()).to_4x4()
G = Matrix.Translation(Vector(ref_pos.mean(0))) @ C @ Matrix.Translation(-Vector(pos0.mean(0)))
action = bpy.data.actions.new("Fit")
arm.animation_data.action = action
for i in range(n_out):
    f = p0 + round(i * (p1 - p0) / max(n_out - 1, 1))
    world = {n: G @ m for n, m in pose_at(f).items()}
    for pb in bones:
        if pb.parent:
            p = pb.parent.name
            basis = (rest[p].inverted() @ rest[pb.name]).inverted() @ world[p].inverted() @ world[pb.name]
        else:
            basis = rest[pb.name].inverted() @ world[pb.name]
        pb.location, pb.rotation_quaternion = basis.to_translation(), basis.to_quaternion()
        pb.keyframe_insert("location", frame=i)
        pb.keyframe_insert("rotation_quaternion", frame=i)
bpy.data.objects.remove(pred)

# Render from the saved reference camera with a transparent background.
cam_data = bpy.data.cameras.new("Cam")
cam_data.lens, cam_data.sensor_width, cam_data.sensor_fit = cam_info["lens"], cam_info["sensor_width"], "HORIZONTAL"
cam = bpy.data.objects.new("Cam", cam_data)
cam.matrix_world = Matrix(cam_info["cam_to_world"])
scene.collection.objects.link(cam)
scene.camera = cam
sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", "SUN"))
sun.data.energy = 3
sun.rotation_euler = (0.7, 0.35, 1.05)
scene.collection.objects.link(sun)
scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_x, scene.render.resolution_y = cam_info["width"], cam_info["height"]
scene.render.film_transparent = True
scene.view_settings.view_transform = "Standard"
scene.frame_start, scene.frame_end = 0, n_out - 1
scene.render.filepath = os.path.join(out, "")
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out, "fit.blend"))
bpy.ops.render.render(animation=True)
print("rendered", n_out, "frames to", out)

# Render the reference frame of a rigged GLB from a fixed side camera.
# Usage: python pipeline/render_ref.py <in.glb> <out_dir> [action] [frame] [--split]
# --split: a split screen instead, three-quarter views from the front right (left panel) and front left
# (right panel), so the video shows the creature from two angles at once (the creature faces -Y, the
# glTF convention). The two copies face away from each other, so forward moves don't meet at the seam.
# Writes ref.png and camera.json (intrinsics, camera matrix, 2D joint positions). With --split,
# camera.json also lists both views, each with intrinsics in full-frame pixels and its x range.
import bpy, sys, json, os, math
import cv2, numpy as np
from mathutils import Vector, Matrix
from bpy_extras.object_utils import world_to_camera_view

sys.path.append(os.path.dirname(__file__))
import rig

split = "--split" in sys.argv
glb, out = rig.args()[:2]
rest = [a for a in rig.args()[2:] if a != "--split"]
action = rest[0] if rest else None
frame = int(rest[1]) if len(rest) > 1 else 0
W, H = 1024, 576  # LTX needs sizes divisible by 32
os.makedirs(out, exist_ok=True)

arm, _ = rig.load(glb, out)
scene = bpy.context.scene
arm.animation_data.action = bpy.data.actions[action] if action else None
if not action:
    for pb in arm.pose.bones:
        pb.matrix_basis.identity()
scene.frame_set(frame)

# Bounds of the posed mesh, used to frame the camera.
deps = bpy.context.evaluated_depsgraph_get()
pts = []
for o in scene.objects:
    if o.type == "MESH" and not o.hide_render:
        ev = o.evaluated_get(deps)
        pts += [ev.matrix_world @ v.co for v in ev.to_mesh().vertices]
lo = Vector((min(p[i] for p in pts) for i in range(3)))
hi = Vector((max(p[i] for p in pts) for i in range(3)))
center, size = (lo + hi) / 2, (hi - lo)

# Side camera on +X looking at -X, or the two three-quarter cameras. Leave room so the action can
# leave the rest bounds.
PW = W // 2 if split else W  # panel width
cam_data = bpy.data.cameras.new("Cam")
cam_data.lens, cam_data.sensor_width, cam_data.sensor_fit = 50, 36, "HORIZONTAL"
cam_data.clip_start, cam_data.clip_end = 0.01 * max(size), 100 * max(size)  # any model scale
cam = bpy.data.objects.new("Cam", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam
fov = 2 * math.atan(18 / 50)
views = [((1, 0, 0), size.y, size.x, (math.pi / 2, 0, math.pi / 2))]
if split:
    s = max(size.x, size.y)
    views = [((math.sqrt(.5), -math.sqrt(.5), 0), s, s, (math.pi / 2, 0, math.pi / 4)),
             ((-math.sqrt(.5), -math.sqrt(.5), 0), s, s, (math.pi / 2, 0, -math.pi / 4))]

sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", "SUN"))
sun.data.energy = 3
sun.rotation_euler = (math.radians(40), math.radians(20), math.radians(60))
scene.collection.objects.link(sun)

scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_x, scene.render.resolution_y = PW, H
scene.render.resolution_percentage = 100
scene.view_settings.view_transform = "Standard"
f = cam_data.lens / cam_data.sensor_width * PW
def shoot():
    # Render every view; returns the image files and the cameras.
    panels, cams = [], []
    for i, (d, width, depth, rot) in enumerate(views):
        extent = max(width, size.z * PW / H) * 1.5
        cam.location = center + Vector(d) * (extent / 2 / math.tan(fov / 2) + depth / 2)
        cam.rotation_euler = rot
        scene.render.filepath = os.path.join(out, f"view{i}.png")
        bpy.ops.render.render(write_still=True)
        panels.append(scene.render.filepath)
        bpy.context.view_layer.update()
        x0 = i * PW
        cams.append({"K": [[f, 0, x0 + PW / 2], [0, f, H / 2], [0, 0, 1]], "cam_to_world": [list(r) for r in cam.matrix_world],
                     "box": [x0, x0 + PW]})  # Blender camera: -Z forward, +Y up
    return panels, cams


# A plain backdrop that the creature stands out from, so masks separate cleanly in the video: the
# first of gray, chroma green, blue and magenta (linear colours) that leaves (almost) the most
# creature pixels clearly different from it, judged on a render with no background.
BACKDROPS = [(0.6, 0.6, 0.6), (0.1, 0.6, 0.15), (0.1, 0.2, 0.8), (0.7, 0.1, 0.6)]
scene.render.film_transparent = True
px = np.concatenate([(lambda im: im[im[..., 3] > 127][:, 2::-1])(cv2.imread(p, cv2.IMREAD_UNCHANGED)) for p in shoot()[0]]).astype(int)
srgb = lambda c: [255 * (1.055 * x ** (1 / 2.4) - 0.055) for x in c]
share = [(np.abs(px - srgb(c)).sum(1) > 90).mean() for c in BACKDROPS]
bg = next(c for c, s in zip(BACKDROPS, share) if s >= max(share) - 0.02)
scene.render.film_transparent = False
rig.backdrop(bg)
panels, cams = shoot()
bpy.context.scene.camera = cam
joints = {}
cam.matrix_world = Matrix(cams[0]["cam_to_world"])
bpy.context.view_layer.update()
for pb in arm.pose.bones:
    p = world_to_camera_view(scene, cam, arm.matrix_world @ pb.head)
    joints[pb.name] = [p.x * PW, (1 - p.y) * H]
cv2.imwrite(os.path.join(out, "ref.png"), np.hstack([cv2.imread(p) for p in panels]))
for p in panels:
    os.remove(p)
json.dump({
    "width": W, "height": H,
    "K": cams[0]["K"], "cam_to_world": cams[0]["cam_to_world"],
    "lens": cam_data.lens, "sensor_width": cam_data.sensor_width, "clip": [cam_data.clip_start, cam_data.clip_end],
    "action": action, "frame": frame, "background": list(bg),
    "joints2d": joints,
    **({"views": cams} if split else {}),
}, open(os.path.join(out, "camera.json"), "w"), indent=1)
print("wrote", out)

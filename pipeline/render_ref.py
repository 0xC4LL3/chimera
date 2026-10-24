# Render the reference frame of a rigged GLB from a fixed side camera.
# Usage: python pipeline/render_ref.py <in.glb> <out_dir> [action] [frame] [--split]
# --split: a split screen instead, side view on the left and front view on the right, so the video
# shows the creature from two angles at once (the creature faces -Y, the glTF convention).
# Writes ref.png and camera.json (intrinsics, camera matrix, 2D joint positions). With --split,
# camera.json also lists both views, each with intrinsics in full-frame pixels and its x range.
import bpy, sys, json, os, math
import cv2, numpy as np
from mathutils import Vector
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

arm, _ = rig.load(glb)
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

# Side camera on +X looking at -X (and front camera on -Y looking at +Y). Leave room so the action
# can leave the rest bounds.
PW = W // 2 if split else W  # panel width
cam_data = bpy.data.cameras.new("Cam")
cam_data.lens, cam_data.sensor_width, cam_data.sensor_fit = 50, 36, "HORIZONTAL"
cam = bpy.data.objects.new("Cam", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam
fov = 2 * math.atan(18 / 50)
views = [((1, 0, 0), size.y, size.x, (math.pi / 2, 0, math.pi / 2))]
if split:
    views.append(((0, -1, 0), size.x, size.y, (math.pi / 2, 0, 0)))

# Plain background that contrasts with the creature, so masks separate cleanly in the video:
# neutral gray for a colourful creature, chroma green for a grayish one.
area, total = 0.0, Vector((0, 0, 0))
for o in scene.objects:
    if o.type == "MESH":
        for poly in o.data.polygons:
            m = o.material_slots[poly.material_index].material if o.material_slots else None
            bsdf = m and m.node_tree and m.node_tree.nodes.get("Principled BSDF")
            if bsdf:
                total += Vector(bsdf.inputs["Base Color"].default_value[:3]) * poly.area
                area += poly.area
srgb = [c ** (1 / 2.2) for c in total / max(area, 1e-8)]
saturation = (max(srgb) - min(srgb)) / max(max(srgb), 1e-8)
bg = (0.6, 0.6, 0.6) if saturation > 0.25 else (0.1, 0.6, 0.15)
rig.backdrop(bg)
sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", "SUN"))
sun.data.energy = 3
sun.rotation_euler = (math.radians(40), math.radians(20), math.radians(60))
scene.collection.objects.link(sun)

scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_x, scene.render.resolution_y = PW, H
scene.render.resolution_percentage = 100
scene.view_settings.view_transform = "Standard"
f = cam_data.lens / cam_data.sensor_width * PW
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
    if i == 0:
        joints = {}
        for pb in arm.pose.bones:
            p = world_to_camera_view(scene, cam, arm.matrix_world @ pb.head)
            joints[pb.name] = [p.x * PW, (1 - p.y) * H]
cv2.imwrite(os.path.join(out, "ref.png"), np.hstack([cv2.imread(p) for p in panels]))
for p in panels:
    os.remove(p)
json.dump({
    "width": W, "height": H,
    "K": cams[0]["K"], "cam_to_world": cams[0]["cam_to_world"],
    "lens": cam_data.lens, "sensor_width": cam_data.sensor_width,
    "action": action, "frame": frame, "background": list(bg),
    "joints2d": joints,
    **({"views": cams} if split else {}),
}, open(os.path.join(out, "camera.json"), "w"), indent=1)
print("wrote", out)

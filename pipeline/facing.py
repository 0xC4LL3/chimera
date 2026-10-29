# Find the turn (about the vertical, and a shift across) that makes a creature face -Y with its mirror
# plane at x=0, which the cameras and the fitter's symmetry assume. rig.load reads it from the run folder.
# Usage: python pipeline/facing.py <in model> <run_dir>   (writes <run_dir>/facing.json)
# The mirror plane is the vertical plane through the mesh's centre that best maps the mesh onto itself.
# The front is the one of the four views along and across that plane that an image-text model
# (SigLIP 2) finds most like a front view and least like a back view: no rule on the shape alone tells
# a head from a tail, a shell or a backpack for every anatomy. A model with the custom property
# chimera_facing = "as_is" (glTF extras) is kept as modelled.
import bpy, sys, os, math, json
import numpy as np, torch
from mathutils import Matrix, Vector
from PIL import Image
from scipy.spatial import cKDTree
from transformers import AutoModel, AutoProcessor

sys.path.append(os.path.dirname(__file__))
import rig

MODEL = "google/siglip2-so400m-patch14-384"
TEXTS = ["a 3D render of a creature seen from the front, facing the camera, its face and eyes visible",
         "a 3D render of a creature seen from behind, its back to the camera"]

model, run = rig.args()[:2]
rig.load(model)
if any(o.get("chimera_facing") == "as_is" for o in bpy.data.objects):
    # The model says it already faces -Y with its mirror plane at x=0: a machine, whose near-symmetries
    # (a trebuchet's frame front to back) could fool the search below.
    json.dump({"pre": [float(i == j) for i in range(4) for j in range(4)], "scores": []}, open(os.path.join(run, "facing.json"), "w"))
    sys.exit()
deps = bpy.context.evaluated_depsgraph_get()
pts = []
for o in bpy.data.objects:
    if o.type == "MESH":
        ev = o.evaluated_get(deps)
        pts += [ev.matrix_world @ v.co for v in ev.to_mesh().vertices]
pts = np.array(pts)
box = Vector(((pts.min(0) + pts.max(0)) / 2).tolist())
pts = pts[::max(1, len(pts) // 2000)]
center = pts.mean(0)
p = pts - center
size = np.linalg.norm(p.max(0) - p.min(0))
tree = cKDTree(p)


def mismatch(a):
    n = np.array([math.cos(a), math.sin(a), 0])
    return tree.query(p - 2 * np.outer(p @ n, n))[0].mean() / size


angles = np.radians(np.arange(0, 180, 1.0))
a = angles[int(np.argmin([mismatch(a) for a in angles]))]
snap = round(a / (math.pi / 2)) * (math.pi / 2)
a = snap if abs(a - snap) < math.radians(3) else a
n = np.array([math.cos(a), math.sin(a), 0])
f = np.array([-n[1], n[0], 0])
dirs = [f, -f, n, -n]

# Render the creature from each direction, looking back at it.
scene = bpy.context.scene
world = bpy.data.worlds.new("W")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.6, 0.6, 0.6, 1)
scene.world = world
sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", "SUN"))
sun.data.energy = 3
sun.rotation_euler = (0.7, 0.35, 1.05)
scene.collection.objects.link(sun)
cam = bpy.data.objects.new("Cam", bpy.data.cameras.new("Cam"))
cam.data.type, cam.data.ortho_scale = "ORTHO", 1.1 * size
cam.data.clip_start, cam.data.clip_end = 0.01 * size, 10 * size
scene.collection.objects.link(cam)
scene.camera = cam
scene.render.engine = "CYCLES"
prefs = bpy.context.preferences.addons["cycles"].preferences
prefs.compute_device_type = "CUDA"
prefs.get_devices()
for d in prefs.devices:
    d.use = d.type == "CUDA"
scene.cycles.device, scene.cycles.samples, scene.cycles.use_denoising = "GPU", 16, False
scene.render.resolution_x = scene.render.resolution_y = 384
scene.view_settings.view_transform = "Standard"
images = []
for k, d in enumerate(dirs):
    across = np.cross(d, [0, 0, 1])
    cam.data.ortho_scale = 1.1 * max(np.ptp(p @ across), np.ptp(p[:, 2]))  # fill the view: a long body seen end-on is small
    cam.location = box + Vector((d * 2 * size).tolist())
    cam.rotation_euler = Vector((-d).tolist()).to_track_quat("-Z", "Y").to_euler()
    scene.render.filepath = os.path.join(run, f"facing_{k}.png")
    bpy.ops.render.render(write_still=True)
    images.append(Image.open(scene.render.filepath).convert("RGB"))

model = AutoModel.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()
proc = AutoProcessor.from_pretrained(MODEL)
inputs = proc(text=TEXTS, images=images, padding="max_length", max_length=64, return_tensors="pt").to("cuda")
with torch.no_grad():
    logits = model(**inputs).logits_per_image.float().cpu().numpy()
scores = logits[:, 0] - logits[:, 1]
front = dirs[int(scores.argmax())]

turn = Matrix.Rotation(-math.pi / 2 - math.atan2(front[1], front[0]), 4, "Z")
x = (turn @ Matrix.Translation(center)).translation.x
pre = Matrix.Translation((-x, 0, 0)) @ turn
json.dump({"pre": [v for row in pre for v in row], "scores": scores.round(2).tolist()},
          open(os.path.join(run, "facing.json"), "w"))
print("front view", int(scores.argmax()), "scores", scores.round(2).tolist())

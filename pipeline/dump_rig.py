# Dump a rig to plain arrays for the torch fitter.
# Usage: python pipeline/dump_rig.py <in.glb> <run_dir>
# Reads <run_dir>/camera.json for the reference action/frame, writes <run_dir>/rig.npz:
# names, parents, rest (B,4,4), ref_basis (B,4,4), verts (V,3), faces (F,3), weights (V,B),
# ref_verts (V,3) = Blender's own skinned mesh in the reference pose, for checking, and pre (4,4) =
# the turn rig.load gave the creature.
# Meshes over MAX_TRIS triangles in all are simplified first, keeping their skin weights: the capture
# (fit, cleanup, check) runs on this stand-in, while renders and the export use the full model, whose
# motion lives in the bones.
import bpy, bmesh, sys, os, json
import numpy as np

sys.path.append(os.path.dirname(__file__))
import rig

MAX_TRIS = 50000

glb, run = rig.args()[:2]
cam = json.load(open(os.path.join(run, "camera.json")))
arm, _ = rig.load(glb, run)
meshes = [o for o in bpy.data.objects if o.type == "MESH"]
for o in meshes:
    o.data.calc_loop_triangles()
total = sum(len(o.data.loop_triangles) for o in meshes)
if total > MAX_TRIS:
    for o in meshes:
        o.shape_key_clear()  # the capture never moves shape keys, and Decimate refuses meshes that have them
        bm = bmesh.new()  # weld first: some exporters (SkinTokens) leave every triangle's corners separate
        bm.from_mesh(o.data)
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6 * max(o.dimensions))
        bm.to_mesh(o.data)
        bm.free()
        bpy.context.view_layer.objects.active = o
        m = o.modifiers.new("Simplify", "DECIMATE")
        m.ratio = MAX_TRIS / total
        bpy.ops.object.modifier_move_to_index(modifier=m.name, index=0)
        bpy.ops.object.modifier_apply(modifier=m.name)
    print(f"simplified {total} triangles to {sum(len(o.data.polygons) for o in meshes)} faces")
bones = list(arm.pose.bones)
names = [pb.name for pb in bones]
idx = {n: i for i, n in enumerate(names)}

verts, faces, weights = [], [], []
for o in bpy.data.objects:
    if o.type != "MESH":
        continue
    o.data.calc_loop_triangles()
    off = sum(len(v) for v in verts)
    groups = {g.index: g.name for g in o.vertex_groups}
    w = np.zeros((len(o.data.vertices), len(names)))
    for v in o.data.vertices:
        for g in v.groups:
            if groups[g.group] in idx:
                w[v.index, idx[groups[g.group]]] = g.weight
    verts.append(np.array([o.matrix_world @ v.co for v in o.data.vertices]))
    faces.append(np.array([t.vertices[:] for t in o.data.loop_triangles]) + off)
    weights.append(w / np.maximum(w.sum(1, keepdims=True), 1e-8))

if cam["action"]:  # otherwise the reference is the rest pose rig.load leaves in place
    arm.animation_data.action = bpy.data.actions[cam["action"]]
    bpy.context.scene.frame_set(cam["frame"])
deps = bpy.context.evaluated_depsgraph_get()
ref_verts = []
for o in bpy.data.objects:
    if o.type == "MESH":
        ev = o.evaluated_get(deps)
        ref_verts.append(np.array([ev.matrix_world @ v.co for v in ev.to_mesh().vertices]))

ref_basis = [pb.matrix_basis.copy() for pb in bones]

np.savez(os.path.join(run, "rig.npz"),
         names=np.array(names),
         parents=np.array([idx[pb.parent.name] if pb.parent else -1 for pb in bones]),
         rest=np.array([b.matrix_local for b in arm.data.bones]),
         ref_basis=np.array(ref_basis),
         verts=np.concatenate(verts), faces=np.concatenate(faces),
         weights=np.concatenate(weights), ref_verts=np.concatenate(ref_verts),
         pre=np.array(arm["pre"]).reshape(4, 4))
print("dumped", len(names), "bones", sum(len(v) for v in verts), "verts")

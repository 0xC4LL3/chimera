# Dump a rig to plain arrays for the torch fitter.
# Usage: blender -b --python proto/dump_rig.py -- <in.glb> <run_dir>
# Reads <run_dir>/camera.json for the reference action/frame, writes <run_dir>/rig.npz:
# names, parents, rest (B,4,4), ref_basis (B,4,4), verts (V,3), faces (F,3), weights (V,B),
# ref_verts (V,3) = Blender's own skinned mesh in the reference pose, for checking.
import bpy, sys, os, json
import numpy as np

sys.path.append(os.path.dirname(__file__))
import rig

glb, run = sys.argv[sys.argv.index("--") + 1:][:2]
cam = json.load(open(os.path.join(run, "camera.json")))
arm, _ = rig.load(glb)
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

arm.animation_data.action = bpy.data.actions[cam["action"]]
bpy.context.scene.frame_set(cam["frame"])
deps = bpy.context.evaluated_depsgraph_get()
ref_verts = []
for o in bpy.data.objects:
    if o.type == "MESH":
        ev = o.evaluated_get(deps)
        ref_verts.append(np.array([ev.matrix_world @ v.co for v in ev.to_mesh().vertices]))

np.savez(os.path.join(run, "rig.npz"),
         names=np.array(names),
         parents=np.array([idx[pb.parent.name] if pb.parent else -1 for pb in bones]),
         rest=np.array([b.matrix_local for b in arm.data.bones]),
         ref_basis=np.array([pb.matrix_basis for pb in bones]),
         verts=np.concatenate(verts), faces=np.concatenate(faces),
         weights=np.concatenate(weights), ref_verts=np.concatenate(ref_verts))
print("dumped", len(names), "bones", sum(len(v) for v in verts), "verts")

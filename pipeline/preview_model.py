# A GLB preview of any model the pipeline reads (GLB, glTF or FBX) and what the app needs to know
# about it: how many bones it has (none: it will be auto-rigged) and its animations, named as
# rig.load names them, so a starting pose picked from the list is found later.
# A model with a rig also gets skeleton.glb: its bones as solid shapes, the way Blender draws them,
# inside a see-through body, so the user can see the rig they supplied or the one auto-rigging made.
# Usage: python pipeline/preview_model.py <in model> <out_dir>   (writes preview.glb, info.json, skeleton.glb)
import bpy, os, json
from mathutils import Vector
import rig

src, out = rig.args()[:2]
os.makedirs(out, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
rig.import_model(src)
rig.drop_missing_textures(src)
bones = sum(len(o.data.bones) for o in bpy.data.objects if o.type == "ARMATURE")
moves_bones = lambda a: any(fc.data_path.startswith("pose.bones") for l in a.layers for s in l.strips
                            for cb in s.channelbags for fc in cb.fcurves)
info = {"bones": bones, "animations": [a.name for a in bpy.data.actions if moves_bones(a)]}
bpy.ops.export_scene.gltf(filepath=os.path.join(out, "preview.glb"), export_format="GLB", export_animations=False)
json.dump(info, open(os.path.join(out, "info.json"), "w"))
print(json.dumps(info))


def material(name, color, alpha=1.0, glow=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1)
    bsdf.inputs["Alpha"].default_value = alpha
    bsdf.inputs["Emission Color"].default_value = (*color, 1)
    bsdf.inputs["Emission Strength"].default_value = glow
    return m


if bones:
    ghost = material("body", (0.6, 0.62, 0.66), alpha=0.15)
    for o in bpy.data.objects:
        if o.type == "MESH":
            o.data.materials.clear()
            o.data.materials.append(ghost)
    bpy.context.view_layer.update()  # world matrices of imported parents are stale until now
    heads = {(a.name, b.name): a.matrix_world @ b.head_local for a in bpy.data.objects if a.type == "ARMATURE" for b in a.data.bones}
    lo = Vector([min(p[i] for p in heads.values()) for i in range(3)])
    size = (Vector([max(p[i] for p in heads.values()) for i in range(3)]) - lo).length
    verts, faces = [], []

    def octahedron(a, b, width):  # pointing from a to b, widest a fifth of the way along
        d = b - a
        if d.length < 1e-6:
            return
        u = d.orthogonal().normalized() * width
        v = d.normalized().cross(u)
        n = len(verts)
        verts.extend([a, *(a + d * 0.2 + w for w in (u, v, -u, -v)), b])
        faces.extend([(n, n + 1 + i, n + 1 + (i + 1) % 4) for i in range(4)] + [(n + 5, n + 1 + (i + 1) % 4, n + 1 + i) for i in range(4)])

    # Only bones that move mesh, or lead to bones that do: control bones (IK targets and poles) often sit far outside.
    weighted = {o.vertex_groups[g.group].name for o in bpy.data.objects if o.type == "MESH" for v in o.data.vertices
                for g in v.groups if g.weight > 0}
    shown = {(a, n) for (a, n) in heads
             if any(c.name in weighted for c in [bpy.data.objects[a].data.bones[n], *bpy.data.objects[a].data.bones[n].children_recursive])}
    # glTF stores joints, not bone tails (Blender only guesses those), so draw each bone from its parent's joint to its own.
    for (arm, name), head in heads.items():
        b = bpy.data.objects[arm].data.bones[name]
        if (arm, name) not in shown:
            continue
        if b.parent and (arm, b.parent.name) in shown:
            parent = heads[(arm, b.parent.name)]
            octahedron(parent, head, min(0.12 * (head - parent).length, 0.022 * size))
        r = Vector((0, 0, 0.012 * size))  # a small diamond on every joint, leaves included
        octahedron(head - r, head + r, 0.012 * size)
    mesh = bpy.data.meshes.new("bones")
    mesh.from_pydata([Vector(p) for p in verts], [], faces)
    mesh.materials.append(material("bones", (0.9, 0.18, 0.02)))
    bpy.context.scene.collection.objects.link(bpy.data.objects.new("bones", mesh))
    bpy.ops.export_scene.gltf(filepath=os.path.join(out, "skeleton.glb"), export_format="GLB", export_animations=False)

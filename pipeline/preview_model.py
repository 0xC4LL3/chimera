# A GLB preview of any model the pipeline reads (GLB, glTF or FBX) and what the app needs to know
# about it: how many bones it has (none: it will be auto-rigged) and its animations, named as
# rig.load names them, so a starting pose picked from the list is found later.
# Usage: python pipeline/preview_model.py <in model> <out_dir>   (writes preview.glb and info.json)
import bpy, os, json
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

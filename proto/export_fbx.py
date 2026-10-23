# Export a rigged GLB into the FBX layout MoCapAnything's custom-rig preprocessing expects:
# <out>/<Name>/<Name>.fbx (mesh + skeleton) and <Name>-<Action>.fbx per animation clip.
# Usage: blender -b --python proto/export_fbx.py -- <in.glb> <out_root> <Name>
import bpy, sys, os
from mathutils import Matrix

sys.path.append(os.path.dirname(__file__))
import rig

glb, out, name = sys.argv[sys.argv.index("--") + 1:]
d = os.path.join(out, name)
os.makedirs(d, exist_ok=True)

arm, actions = rig.load(glb)

# Match Truebones conventions, which the pipeline assumes: armature data in centimetres under
# an object scale of 0.01. Its BVH export reads armature space and multiplies by 0.01, while
# its OBJ export is in world space, so both end up in metres.
rig.bake(arm, Matrix.Scale(100, 4))
for a in actions:
    for l in a.layers:
        for s in l.strips:
            for cb in s.channelbags:
                for fc in cb.fcurves:
                    if fc.data_path.endswith(".location"):
                        for k in fc.keyframe_points:
                            k.co.y *= 100
                            k.handle_left.y *= 100
                            k.handle_right.y *= 100
arm.scale = (0.01, 0.01, 0.01)


def export(path, action):
    arm.animation_data.action = action
    bpy.context.scene.frame_start, bpy.context.scene.frame_end = map(int, action.frame_range)
    bpy.ops.export_scene.fbx(filepath=path, add_leaf_bones=False, bake_anim=True,
                             bake_anim_use_nla_strips=False, bake_anim_use_all_actions=False)


# The base FBX holds a two-frame rest-pose action because MoCapAnything's fix_fbx needs one.
rest = bpy.data.actions.new("Rest")
arm.animation_data.action = rest
for pb in arm.pose.bones:
    pb.matrix_basis.identity()
    for f in (0, 1):
        pb.keyframe_insert("rotation_quaternion", frame=f)
        pb.keyframe_insert("location", frame=f)
export(os.path.join(d, f"{name}.fbx"), rest)
for a in actions:
    export(os.path.join(d, f"{name}-{a.name.split('|')[-1].replace('_', '')}.fbx"), a)
print("exported", os.listdir(d))

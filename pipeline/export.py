# Export a cleaned animation onto the user's original rig as GLB, FBX and BVH.
# Usage: blender -b --python pipeline/export.py -- <in.glb> <run_dir> <clean.npz> <out_dir> <clip_name>
# The pipeline works on a rebaked copy of the rig whose bones may be oriented differently, so the
# motion is carried over as each bone's world-space deformation (posed matrix times inverse rest),
# which does not depend on bone orientation, onto a fresh import of the user's file with its
# transforms and bone orientations untouched. Both imports take the glTF node pose as rest, so
# the deformations mean the same thing on both.
# Writes <clip_name>.glb/.fbx with two actions (<clip_name> with root motion, <clip_name>_inplace)
# and <clip_name>.bvh with root motion.
import bpy, sys, os
import numpy as np
from mathutils import Matrix

sys.path.append(os.path.dirname(__file__))
import rig as rigmod

glb, run, clean, out, clip = rigmod.args()[:5]
os.makedirs(out, exist_ok=True)
FPS = 24

rig = np.load(os.path.join(run, "rig.npz"))
names, parents, rest = rig["names"].tolist(), rig["parents"], rig["rest"]
local_rest = [rest[i] if p < 0 else np.linalg.inv(rest[p]) @ rest[i] for i, p in enumerate(parents)]
data = np.load(clean)


def deformations(basis):
    # Per frame and bone: posed world matrix times inverse rest, on the rebaked rig (identity object).
    result = []
    for frame in basis:
        mats = []
        for i, p in enumerate(parents):
            mats.append(local_rest[i] @ frame[i] if p < 0 else mats[p] @ local_rest[i] @ frame[i])
        result.append({n: Matrix((m @ np.linalg.inv(rest[i])).tolist()) for i, (n, m) in enumerate(zip(names, mats))})
    return result


bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb, guess_original_bind_pose=False)
scene = bpy.context.scene
scene.render.fps = FPS
arm = next(o for o in scene.objects if o.type == "ARMATURE")
for t in list(arm.animation_data.nla_tracks):
    arm.animation_data.nla_tracks.remove(t)
for a in list(bpy.data.actions):
    bpy.data.actions.remove(a)
A = arm.matrix_world
rest_arm = {b.name: b.matrix_local.copy() for b in arm.data.bones}

actions = []
for name, key in ((clip, "basis"), (clip + "_inplace", "basis_inplace")):
    action = bpy.data.actions.new(name)
    arm.animation_data.action = action
    for f, D in enumerate(deformations(data[key])):
        pose = {}  # armature-space posed matrix per bone of the original rig
        for pb in arm.pose.bones:
            pose[pb.name] = A.inverted() @ D[pb.name] @ A @ rest_arm[pb.name]
            if pb.parent:
                p = pb.parent.name
                basis = (rest_arm[p].inverted() @ rest_arm[pb.name]).inverted() @ pose[p].inverted() @ pose[pb.name]
            else:
                basis = rest_arm[pb.name].inverted() @ pose[pb.name]
            pb.rotation_mode = "QUATERNION"
            pb.location, pb.rotation_quaternion = basis.to_translation(), basis.to_quaternion()
            pb.keyframe_insert("location", frame=f)
            pb.keyframe_insert("rotation_quaternion", frame=f)
    action.use_fake_user = True
    actions.append(action)
scene.frame_start, scene.frame_end = 0, len(data["basis"]) - 1

base = os.path.join(out, clip)
bpy.ops.export_scene.gltf(filepath=base + ".glb", export_format="GLB", export_animation_mode="ACTIONS")
bpy.ops.export_scene.fbx(filepath=base + ".fbx", add_leaf_bones=False, bake_anim=True,
                         bake_anim_use_all_actions=True, bake_anim_use_nla_strips=False, bake_anim_simplify_factor=0)
arm.animation_data.action = actions[0]
bpy.ops.object.select_all(action="DESELECT")
arm.select_set(True)
bpy.context.view_layer.objects.active = arm
bpy.ops.export_anim.bvh(filepath=base + ".bvh", frame_start=scene.frame_start, frame_end=scene.frame_end)
print("exported", os.listdir(out))

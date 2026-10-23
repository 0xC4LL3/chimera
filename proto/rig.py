# Shared Blender helper: load a rigged GLB with object transforms baked into the data,
# so armature space equals world space and the skeleton and mesh share one unit scale.
# Animations are re-keyed from their recorded world-space pose, because applying a
# rotation to an armature recomputes bone rolls and would invalidate the original keys.
import bpy
from mathutils import Matrix


def unscaled(m):
    loc, rot, _ = m.decompose()
    return Matrix.Translation(loc) @ rot.to_matrix().to_4x4()


def load(glb):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=glb, guess_original_bind_pose=False)  # rest = glTF node pose
    scene = bpy.context.scene
    for o in list(bpy.data.objects):
        if (o.type == "MESH" and o.parent is None) or (o.type == "EMPTY" and o.name.endswith("_end")):
            bpy.data.objects.remove(o)  # stray meshes and glTF leaf markers
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    for t in list(arm.animation_data.nla_tracks):
        arm.animation_data.nla_tracks.remove(t)
    actions = list(bpy.data.actions)
    bones = list(arm.pose.bones)  # parents come before children

    # Record the world-space pose of every bone on every frame of every action.
    recorded = {}
    for a in actions:
        arm.animation_data.action = a
        for pb in bones:
            pb.matrix_basis.identity()  # channels the clip doesn't key stay at rest
        f0, f1 = map(int, a.frame_range)
        recorded[a] = {}
        for f in range(f0, f1 + 1):
            scene.frame_set(f)
            recorded[a][f] = {pb.name: unscaled(arm.matrix_world @ pb.matrix) for pb in bones}
    arm.animation_data.action = None
    for pb in bones:
        pb.matrix_basis.identity()
    scene.frame_set(0)

    # Bake transforms. Bone-parented meshes are detached first and re-attached after.
    bone_kids = []
    for o in bpy.data.objects:
        if o.parent is not None:
            if o.parent_type == "BONE":
                bone_kids.append((o, o.parent_bone))
            mw = o.matrix_world.copy()
            o.parent = None
            o.matrix_world = mw
    bpy.ops.object.select_all(action="DESELECT")
    for o in bpy.data.objects:
        o.select_set(o.type in ("ARMATURE", "MESH"))
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    for o in list(bpy.data.objects):
        if o.type == "EMPTY":
            bpy.data.objects.remove(o)
    kids = [o for o, _ in bone_kids]
    for o in bpy.data.objects:
        if o.type == "MESH" and o not in kids:
            o.parent = arm  # keep the armature modifier link visible in the hierarchy
            o.matrix_parent_inverse.identity()
    bpy.context.view_layer.update()
    for o, b in bone_kids:
        o.parent, o.parent_type, o.parent_bone = arm, "BONE", b
        bpy.context.view_layer.update()
        o.matrix_world = Matrix.Identity(4)

    # Re-key each action from the recorded world pose (armature space is now world space).
    rest = {b.name: b.matrix_local for b in arm.data.bones}
    for a in actions:
        for l in a.layers:
            for s in l.strips:
                for cb in s.channelbags:
                    for fc in list(cb.fcurves):
                        cb.fcurves.remove(fc)
        arm.animation_data.action = a
        for f, pose in recorded[a].items():
            for pb in bones:
                if pb.parent:
                    p = pb.parent.name
                    basis = (rest[p].inverted() @ rest[pb.name]).inverted() @ pose[p].inverted() @ pose[pb.name]
                else:
                    basis = rest[pb.name].inverted() @ pose[pb.name]
                pb.rotation_mode = "QUATERNION"
                pb.location, pb.rotation_quaternion = basis.to_translation(), basis.to_quaternion()
                pb.keyframe_insert("location", frame=f)
                pb.keyframe_insert("rotation_quaternion", frame=f)
    arm.animation_data.action = None
    for pb in bones:
        pb.matrix_basis.identity()
    return arm, actions

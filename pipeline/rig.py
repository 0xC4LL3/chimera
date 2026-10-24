# Shared Blender helper: load a rigged GLB with object transforms baked into the data,
# so armature space equals world space and the skeleton and mesh share one unit scale.
# Given the run folder, the creature is also turned to face -Y with its mirror plane at x=0, as
# facing.py found, which the cameras and the fitter's symmetry assume; export.py turns the motion
# back for the user's file.
# Animations are re-keyed from their recorded world-space pose, because applying a
# rotation to an armature recomputes bone rolls and would invalidate the original keys.
import bpy, os, sys, json, shutil
import cv2, numpy as np
from mathutils import Matrix


def args():
    # Script arguments, whether run as `blender -b --python x.py -- args` or `python x.py args` (bpy wheel).
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]


def unscaled(m):
    loc, rot, _ = m.decompose()
    return Matrix.Translation(loc) @ rot.to_matrix().to_4x4()


def bake(arm, pre=Matrix.Identity(4)):
    # Apply `pre` then all object transforms to the data; meshes end as plain children of arm.
    for o in bpy.data.objects:
        mw = o.matrix_world.copy()
        o.parent = None
        o.matrix_world = pre @ mw
    bpy.ops.object.select_all(action="DESELECT")
    for o in bpy.data.objects:
        o.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    for o in bpy.data.objects:
        if o.type == "MESH":
            o.parent = arm
            o.matrix_parent_inverse.identity()


def load(glb, run=None):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=glb, guess_original_bind_pose=False)  # rest = glTF node pose
    scene = bpy.context.scene
    for o in list(bpy.data.objects):
        if (o.type == "MESH" and o.parent is None) or (o.type == "EMPTY" and o.name.endswith("_end")):
            bpy.data.objects.remove(o)  # stray meshes and glTF leaf markers
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    anim = arm.animation_data_create()  # also for rigs that come without animations
    for t in list(anim.nla_tracks):
        anim.nla_tracks.remove(t)
    actions = list(bpy.data.actions)
    bones = list(arm.pose.bones)  # parents come before children
    anim.action = None
    for pb in bones:
        pb.matrix_basis.identity()
    bpy.context.view_layer.update()
    face = run and os.path.join(run, "facing.json")
    pre = Matrix([json.load(open(face))["pre"][i:i + 4] for i in range(0, 16, 4)]) if face and os.path.exists(face) else Matrix.Identity(4)
    arm["pre"] = [x for row in pre for x in row]  # for undoing the turn on export

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
            recorded[a][f] = {pb.name: unscaled(pre @ arm.matrix_world @ pb.matrix) for pb in bones}
    arm.animation_data.action = None
    for pb in bones:
        pb.matrix_basis.identity()
    scene.frame_set(0)

    # Meshes parented to a bone (eyes, props), directly or through empties, become meshes skinned
    # fully to that bone.
    for o in bpy.data.objects:
        if o.type != "MESH" or any(m.type == "ARMATURE" for m in o.modifiers):
            continue
        p = o
        while p and p.parent_type != "BONE":
            p = p.parent
        if p:
            o.vertex_groups.new(name=p.parent_bone).add(range(len(o.data.vertices)), 1.0, "REPLACE")
            o.modifiers.new("Armature", "ARMATURE").object = arm
    bake(arm, pre)
    for o in list(bpy.data.objects):
        if o.type == "EMPTY":
            bpy.data.objects.remove(o)

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


def backdrop(color):
    # A world that shows `color` behind the creature but lights it neutral gray, so a chroma green
    # backdrop does not tint the creature.
    world = bpy.data.worlds.new("W")
    world.use_nodes = True
    nt = world.node_tree
    light = nt.nodes["Background"]
    light.inputs[0].default_value = (0.6, 0.6, 0.6, 1)
    seen = nt.nodes.new("ShaderNodeBackground")
    seen.inputs[0].default_value = (*color, 1)
    path, mix = nt.nodes.new("ShaderNodeLightPath"), nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(path.outputs["Is Camera Ray"], mix.inputs[0])
    nt.links.new(light.outputs[0], mix.inputs[1])
    nt.links.new(seen.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], nt.nodes["World Output"].inputs[0])
    bpy.context.scene.world = world


def render_from_camera(cam_info, out, n_frames, transparent=True):
    # Render frames 0..n-1 from the saved reference camera, by default with a transparent background;
    # a split-screen camera (render_ref.py --split) renders every view and puts them side by side.
    # Also saves <out>/fit.blend for inspection.
    if cam_info.get("views"):
        dirs = [f"{out}_view{i}" for i in range(len(cam_info["views"]))]
        for d, v in zip(dirs, cam_info["views"]):
            render_from_camera({**cam_info, "cam_to_world": v["cam_to_world"], "width": v["box"][1] - v["box"][0], "views": None},
                               d, n_frames, transparent)
        os.makedirs(out, exist_ok=True)
        for f in sorted(os.listdir(dirs[0])):
            if f.endswith(".png"):
                cv2.imwrite(os.path.join(out, f), np.hstack([cv2.imread(os.path.join(d, f), cv2.IMREAD_UNCHANGED) for d in dirs]))
        for d in dirs:
            shutil.rmtree(d)
        bpy.context.scene.render.resolution_x = cam_info["width"]
        return
    os.makedirs(out, exist_ok=True)
    scene = bpy.context.scene
    cam_data = bpy.data.cameras.new("Cam")
    cam_data.lens, cam_data.sensor_width, cam_data.sensor_fit = cam_info["lens"], cam_info["sensor_width"], "HORIZONTAL"
    cam = bpy.data.objects.new("Cam", cam_data)
    cam.matrix_world = Matrix(cam_info["cam_to_world"])
    scene.collection.objects.link(cam)
    scene.camera = cam
    if not any(o.type == "LIGHT" for o in scene.objects):
        sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", "SUN"))
        sun.data.energy = 3
        sun.rotation_euler = (0.7, 0.35, 1.05)
        scene.collection.objects.link(sun)
    # Cycles on the GPU: Eevee falls back to software OpenGL under WSL (~10 s per frame).
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "CUDA"
    prefs.get_devices()
    for d in prefs.devices:
        d.use = d.type == "CUDA"
    # Few samples are enough for a check render; persistent data skips rebuilding the scene per frame.
    scene.cycles.device, scene.cycles.samples, scene.cycles.use_denoising = "GPU", 4, False
    scene.render.use_persistent_data = True
    scene.render.resolution_x, scene.render.resolution_y = cam_info["width"], cam_info["height"]
    scene.render.film_transparent = transparent
    scene.view_settings.view_transform = "Standard"
    scene.frame_start, scene.frame_end = 0, n_frames - 1
    scene.render.filepath = os.path.join(out, "")
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out, "fit.blend"))
    bpy.ops.render.render(animation=True)
    print("rendered", n_frames, "frames to", out)

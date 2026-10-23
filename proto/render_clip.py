# Render one of the rig's own animations from the reference camera, as a ground-truth test clip.
# Usage: blender -b --python proto/render_clip.py -- <in.glb> <run_dir>
# Uses the action in <run_dir>/camera.json (from render_ref.py), starting at its reference frame.
# Writes <run_dir>/clip/####.png on a plain background and <run_dir>/gt.npz with world joint
# positions per frame.
import bpy, sys, os, json
import numpy as np

sys.path.append(os.path.dirname(__file__))
import rig

glb, run = sys.argv[sys.argv.index("--") + 1:][:2]
cam = json.load(open(os.path.join(run, "camera.json")))
arm, _ = rig.load(glb)
action = bpy.data.actions[cam["action"]]
arm.animation_data.action = action
scene = bpy.context.scene
f0, f1 = cam["frame"], int(action.frame_range[1])

joints = []
for f in range(f0, f1 + 1):
    scene.frame_set(f)
    joints.append([arm.pose.bones[b.name].head[:] for b in arm.data.bones])
np.savez(os.path.join(run, "gt.npz"), joints=np.array(joints), names=[b.name for b in arm.data.bones])
# Shift the action so the reference frame lands on frame 0.
for l in action.layers:
    for s in l.strips:
        for cb in s.channelbags:
            for fc in cb.fcurves:
                for k in fc.keyframe_points:
                    k.co.x -= f0
                    k.handle_left.x -= f0
                    k.handle_right.x -= f0

world = bpy.data.worlds.new("W")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.6, 0.6, 0.6, 1)
scene.world = world
rig.render_from_camera(cam, os.path.join(run, "clip"), f1 - f0 + 1, transparent=False)

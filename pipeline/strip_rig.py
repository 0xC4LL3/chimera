# Keep only a model's meshes, posed by their skeleton's rest pose (the one rig.load uses), so a rigged
# model can be re-rigged. Baking the pose keeps bone-parented parts (eyes) where the body puts them.
# The meshes are joined into one: SkinTokens crashes Blender on meshes under 990 vertices (small eyes).
# Usage: python pipeline/strip_rig.py <in.glb> <out.glb>
import bpy
import rig

src, dst = rig.args()[:2]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src, guess_original_bind_pose=False)
for arm in [o for o in bpy.data.objects if o.type == "ARMATURE"]:
    if arm.animation_data:
        arm.animation_data.action = None
    for pb in arm.pose.bones:
        pb.matrix_basis.identity()
bpy.context.view_layer.update()
for o in [o for o in bpy.data.objects if o.type == "MESH"]:
    bpy.context.view_layer.objects.active = o
    for m in [m for m in o.modifiers if m.type == "ARMATURE"]:
        bpy.ops.object.modifier_apply(modifier=m.name)
    mw = o.matrix_world.copy()
    o.parent = None
    o.matrix_world = mw
    o.vertex_groups.clear()
for o in [o for o in bpy.data.objects if o.type != "MESH"]:
    bpy.data.objects.remove(o)
meshes = list(bpy.data.objects)
for o in meshes:
    o.select_set(True)
bpy.context.view_layer.objects.active = max(meshes, key=lambda o: len(o.data.vertices))
bpy.ops.object.join()
bpy.ops.export_scene.gltf(filepath=dst, export_animations=False)

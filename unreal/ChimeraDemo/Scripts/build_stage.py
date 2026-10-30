"""Import an actual Chimera FBX and build the dedicated engine preview level."""
import json
from pathlib import Path
import unreal

PROJECT = Path(__file__).absolute().parents[1]
REPO = PROJECT.parents[1]
ASSETS = "/Game/Chimera/Hero"
STAGE = "/Game/Chimera/Stage"
LEVEL = STAGE + "/L_Chimera"
tools = unreal.AssetToolsHelpers.get_asset_tools()
actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
levels = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
materials = unreal.MaterialEditingLibrary


def material(name, color, roughness, metallic=0):
    path = STAGE + "/" + name
    if unreal.EditorAssetLibrary.does_asset_exist(path):
        return unreal.load_asset(path)
    mat = tools.create_asset(name, STAGE, unreal.Material, unreal.MaterialFactoryNew())
    rgb = materials.create_material_expression(mat, unreal.MaterialExpressionConstant3Vector, -300, 0)
    rgb.set_editor_property("constant", unreal.LinearColor(*color, 1))
    materials.connect_material_property(rgb, "", unreal.MaterialProperty.MP_BASE_COLOR)
    for prop, value, y in ((unreal.MaterialProperty.MP_ROUGHNESS, roughness, 100),
                           (unreal.MaterialProperty.MP_METALLIC, metallic, 200)):
        scalar = materials.create_material_expression(mat, unreal.MaterialExpressionConstant, -300, y)
        scalar.set_editor_property("r", value)
        materials.connect_material_property(scalar, "", prop)
    materials.recompile_material(mat)
    unreal.EditorAssetLibrary.save_loaded_asset(mat)
    return mat


def shape(name, mesh, location, scale, mat):
    actor = actors.spawn_actor_from_class(unreal.StaticMeshActor, unreal.Vector(*location))
    actor.set_actor_label(name)
    actor.static_mesh_component.set_static_mesh(unreal.load_asset("/Engine/BasicShapes/" + mesh))
    actor.static_mesh_component.set_material(0, mat)
    actor.set_actor_scale3d(unreal.Vector(*scale))
    return actor


def light(name, location, target, intensity, temperature, width):
    rotation = unreal.MathLibrary.find_look_at_rotation(unreal.Vector(*location), unreal.Vector(*target))
    actor = actors.spawn_actor_from_class(unreal.RectLight, unreal.Vector(*location), rotation)
    actor.set_actor_label(name)
    component = actor.get_component_by_class(unreal.RectLightComponent)
    component.set_mobility(unreal.ComponentMobility.MOVABLE)
    component.set_editor_properties(dict(intensity=intensity, intensity_units=unreal.LightUnits.LUMENS,
                                        attenuation_radius=3000, source_width=width, source_height=width,
                                        use_temperature=True, temperature=temperature))


def camera(name, location, target, primary=False):
    rotation = unreal.MathLibrary.find_look_at_rotation(unreal.Vector(*location), unreal.Vector(*target))
    actor = actors.spawn_actor_from_class(unreal.CameraActor, unreal.Vector(*location), rotation)
    actor.set_actor_label(name)
    actor.camera_component.set_editor_properties(dict(field_of_view=38, aspect_ratio=16 / 9, constrain_aspect_ratio=True))
    if primary:
        actor.set_editor_property("auto_activate_for_player", unreal.AutoReceiveInput.PLAYER0)
    return actor


def import_hero(source, animation_hint):
    # This folder contains only this script's generated FBX import.
    if unreal.EditorAssetLibrary.does_directory_exist(ASSETS):
        unreal.EditorAssetLibrary.delete_directory(ASSETS)
    options = unreal.FbxImportUI()
    options.set_editor_properties(dict(import_mesh=True, import_as_skeletal=True, import_animations=True,
                                      import_materials=True, import_textures=True, create_physics_asset=False,
                                      automated_import_should_detect_type=False,
                                      mesh_type_to_import=unreal.FBXImportType.FBXIT_SKELETAL_MESH))
    for data in (options.skeletal_mesh_import_data, options.anim_sequence_import_data):
        data.set_editor_properties(dict(convert_scene=True, convert_scene_unit=True, force_front_x_axis=True))
    options.skeletal_mesh_import_data.set_editor_property("normal_import_method", unreal.FBXNormalImportMethod.FBXNIM_IMPORT_NORMALS)
    options.anim_sequence_import_data.set_editor_properties(dict(use_default_sample_rate=False, custom_sample_rate=24,
                                                                import_bone_tracks=True))
    task = unreal.AssetImportTask()
    task.set_editor_properties(dict(filename=str(source), destination_path=ASSETS, destination_name="Hero",
                                   automated=True, replace_existing=True, replace_existing_settings=True,
                                   save=True, factory=unreal.FbxFactory(), options=options))
    tools.import_asset_tasks([task])
    imported = list(task.get_objects())
    meshes = [asset for asset in imported if isinstance(asset, unreal.SkeletalMesh)]
    if len(meshes) != 1:
        raise RuntimeError(f"Expected one imported skeletal mesh, got {len(meshes)}. Inspect the FBX import log.")
    mesh = meshes[0]
    animations = [asset for asset in imported if isinstance(asset, unreal.AnimSequence)]
    # FBX factories can return only the primary mesh. The imported animations share its skeleton.
    if not animations:
        assets = [unreal.load_asset(path) for path in unreal.EditorAssetLibrary.list_assets(ASSETS, recursive=True)]
        animations = [asset for asset in assets if isinstance(asset, unreal.AnimSequence)
                      and asset.get_editor_property("skeleton") == mesh.get_editor_property("skeleton")]
    choices = sorted((a for a in animations if animation_hint.lower() in a.get_name().lower()), key=lambda a: a.get_name())
    if len(choices) != 1:
        raise RuntimeError(f"Set animation_hint to select one imported animation. Candidates: {[a.get_name() for a in animations]}")
    unreal.EditorAssetLibrary.save_directory(ASSETS, only_if_is_dirty=False, recursive=True)
    return mesh, choices[0], animations


def build():
    config = json.loads((PROJECT / "Config/demo.json").read_text())
    source = REPO / config["source_fbx"]
    if not source.is_file():
        raise FileNotFoundError(source)
    mesh, animation, animations = import_hero(source, config["animation_hint"])
    if unreal.EditorAssetLibrary.does_asset_exist(LEVEL):
        levels.load_level(LEVEL)
        actors.destroy_actors(actors.get_all_level_actors())
    else:
        levels.new_level(LEVEL)

    floor_mat = material("M_Floor", (.027, .033, .039), .8)
    plinth_mat = material("M_Plinth", (.065, .073, .081), .55, .12)
    rim_mat = material("M_Rim", (.40, .115, .035), .38, .3)
    shape("Floor", "Plane", (0, 0, -2), (180, 180, 1), floor_mat)
    shape("Plinth", "Cylinder", (0, 0, 13), (9, 9, .26), plinth_mat)
    shape("Copper rim", "Cylinder", (0, 0, 27), (9.02, 9.02, .02), rim_mat)
    shape("Stage surface", "Cylinder", (0, 0, 29), (8.95, 8.95, .02), plinth_mat)

    hero = actors.spawn_actor_from_class(unreal.SkeletalMeshActor, unreal.Vector(0, 0, 30),
                                          unreal.Rotator(yaw=config["hero_yaw_degrees"]))
    hero.set_actor_label("Chimera exported rig")
    component = hero.skeletal_mesh_component
    component.set_skeletal_mesh_asset(mesh)
    bounds = mesh.get_bounds()
    scale = config["hero_height_cm"] / (2 * bounds.box_extent.z)
    hero.set_actor_scale3d(unreal.Vector(scale, scale, scale))
    component.override_animation_data(animation, is_looping=True, is_playing=True, position=0, play_rate=1)
    component.set_update_animation_in_editor(True)
    origin, extent = hero.get_actor_bounds(False)
    position = hero.get_actor_location()
    hero.set_actor_location(unreal.Vector(position.x - origin.x, position.y - origin.y,
                                          position.z + 30 - (origin.z - extent.z)), False, False)

    target = (0, 0, 155)
    light("Key / warm softbox", (480, -420, 620), target, 5000, 5200, 400)
    light("Fill / cool softbox", (200, 500, 340), target, 1700, 7000, 500)
    light("Rim / gold softbox", (-360, 100, 520), target, 5200, 3800, 260)
    sky = actors.spawn_actor_from_class(unreal.SkyLight, unreal.Vector(0, 0, 400))
    sky.set_actor_label("Studio ambient")
    sky.light_component.set_editor_properties(dict(source_type=unreal.SkyLightSourceType.SLS_SPECIFIED_CUBEMAP,
                                                   cubemap=unreal.load_asset("/Engine/MapTemplates/Sky/DaylightAmbientCubemap"),
                                                   intensity=.35))
    pp = actors.spawn_actor_from_class(unreal.PostProcessVolume, unreal.Vector())
    pp.set_actor_label("Fixed exposure")
    pp.set_editor_property("unbound", True)
    settings = pp.get_editor_property("settings")
    for name, value in dict(auto_exposure_method=unreal.AutoExposureMethod.AEM_MANUAL,
                            auto_exposure_apply_physical_camera_exposure=False, auto_exposure_bias=-4,
                            bloom_intensity=.12, vignette_intensity=.18, motion_blur_amount=0).items():
        settings.set_editor_property("override_" + name, True)
        settings.set_editor_property(name, value)
    pp.set_editor_property("settings", settings)
    observer = actors.spawn_actor_from_class(unreal.PlayerStart, unreal.Vector(0, -2500, 100))
    observer.set_actor_label("Observer start")
    views = [camera("Camera / hero", (720, 820, 360), target, primary=True),
             camera("Camera / side", (0, -1100, 300), target),
             camera("Camera / front", (1100, 0, 300), target)]
    actors.set_selected_level_actors([])
    levels.pilot_level_actor(views[0])
    levels.set_exact_camera_view(True)
    levels.editor_set_game_view(True)
    levels.save_current_level()

    state = {"source_fbx": str(source), "source_label": config["source_label"], "mesh": mesh.get_path_name(),
             "animation": animation.get_path_name(), "animations": [a.get_path_name() for a in animations],
             "duration_seconds": animation.get_play_length(), "bones": component.get_num_bones(),
             "level": LEVEL, "hero_scale": scale}
    (PROJECT / "Saved").mkdir(exist_ok=True)
    (PROJECT / "Saved/demo-state.json").write_text(json.dumps(state, indent=2))
    unreal.log("Chimera engine preview built: " + json.dumps(state))
    return state


if __name__ == "__main__":
    build()
    import sys
    sys.path.insert(0, str(PROJECT / "Scripts"))
    import preview
    preview.start()

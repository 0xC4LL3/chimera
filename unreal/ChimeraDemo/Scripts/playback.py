"""Record native Play In Editor playback evidence from the three stage cameras.

From Saved/command.py: import playback; playback.start()
Playback uses the engine clock at normal speed. The samples are review evidence.
"""
from datetime import datetime
import json
from pathlib import Path
import time
import traceback
import unreal

PROJECT = Path(__file__).absolute().parents[1]
handle = None


def stop():
    global handle
    if handle is not None:
        unreal.unregister_slate_post_tick_callback(handle)
        handle = None
    unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).editor_request_end_play()


def start():
    global handle
    levels = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    if handle is not None or levels.is_in_play_in_editor():
        raise RuntimeError("Finish the current playback session first.")
    state = {"shot": -1, "started": time.monotonic(), "last_sample": -1, "task": None}
    out = PROJECT / "Saved/playback" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    out.mkdir(parents=True)
    proof = json.loads((PROJECT / "Saved/demo-state.json").read_text())
    proof.update(method="Native Play In Editor playback", play_rate=1, shots=[], status="starting")
    animation = unreal.load_asset(proof["animation"])
    shots = ("hero", "side", "front")
    (PROJECT / "Saved/latest-playback.txt").write_text(str(out))

    def save(status):
        proof["status"] = status
        (out / "playback.json").write_text(json.dumps(proof, indent=2))

    def tick(dt):
        try:
            world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_game_world()
            if world is None:
                if time.monotonic() - state["started"] > 60:
                    raise RuntimeError("Play In Editor did not start within 60 seconds.")
                return
            if state["shot"] == -1:
                actors = unreal.GameplayStatics.get_all_actors_of_class(world, unreal.Actor)
                actor = next(a for a in actors if a.get_actor_label() == "Chimera exported rig")
                state["component"] = actor.skeletal_mesh_component
                state["cameras"] = {name: next(a for a in actors if a.get_actor_label() == "Camera / " + name)
                                    for name in shots}
                state["controller"] = unreal.GameplayStatics.get_player_controller(world, 0)
                proof["world"] = world.get_path_name()
                proof["actor"] = actor.get_path_name()
                proof["bone_names"] = [str(state["component"].get_bone_name(i))
                                       for i in range(state["component"].get_num_bones())]
                state["shot"] = 0
                state["begin"] = True
            component = state["component"]
            if state["begin"]:
                name = shots[state["shot"]]
                state["controller"].set_view_target_with_blend(state["cameras"][name], 0)
                component.play_animation(animation, False)
                state["shot_start"] = unreal.GameplayStatics.get_time_seconds(world)
                state["last_sample"] = -1
                state["begin"] = False
                state["screenshot"] = False
                proof["shots"].append({"camera": name, "samples": []})
            elapsed = unreal.GameplayStatics.get_time_seconds(world) - state["shot_start"]
            if elapsed - state["last_sample"] >= .1:
                locations = [component.get_socket_location(name) for name in proof["bone_names"]]
                proof["shots"][-1]["samples"].append({"world_seconds": elapsed,
                    "animation_seconds": component.get_position(), "playing": component.is_playing(),
                    "bone_locations": [[round(v.x, 3), round(v.y, 3), round(v.z, 3)] for v in locations]})
                state["last_sample"] = elapsed
            if not state["screenshot"] and elapsed >= proof["duration_seconds"] / 2:
                state["task"] = unreal.AutomationLibrary.take_high_res_screenshot(
                    1920, 1080, str(out / (shots[state["shot"]] + ".png")), delay=0)
                state["screenshot"] = True
            if elapsed >= proof["duration_seconds"] + .3:
                if state["task"] is not None and not state["task"].is_task_done():
                    return
                state["task"] = None
                state["shot"] += 1
                save("running")
                if state["shot"] == len(shots):
                    save("complete")
                    stop()
                    return
                state["begin"] = True
        except Exception:
            (out / "error.txt").write_text(traceback.format_exc())
            save("failed")
            stop()

    save("starting")
    levels.editor_request_begin_play()
    handle = unreal.register_slate_post_tick_callback(tick)
    unreal.log("Chimera native playback evidence: " + str(out))
    return str(out)

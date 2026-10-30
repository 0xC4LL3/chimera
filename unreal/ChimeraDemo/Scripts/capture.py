"""Render the imported animation at 24 fps from the three saved stage cameras.

From Saved/command.py: import capture; capture.start(component, animations[clip_index])
The animation is sampled by Unreal. This does not create or change keyframes.
"""
from datetime import datetime
import json
import math
from pathlib import Path
import time
import traceback
import unreal

PROJECT = Path(__file__).absolute().parents[1]
handle = None
write_status = None


def stop(cancelled=True):
    global handle
    if handle is not None:
        unreal.unregister_slate_post_tick_callback(handle)
        handle = None
        if cancelled:
            write_status("cancelled")


def start(component, animation, shots=("hero", "side", "front"), fps=24, width=1920, height=1080):
    global handle, write_status
    stop()
    levels = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    if levels.is_in_play_in_editor():
        raise RuntimeError("End Play In Editor before rendering the camera sequences.")
    actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors()
    cameras = {name: next(a for a in actors if a.get_actor_label() == "Camera / " + name) for name in shots}
    out = PROJECT / "Saved/captures" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    for name in shots:
        (out / name).mkdir(parents=True)
    state = {"shot": 0, "frame": 0, "task": None, "phase": "set", "ready": 0}
    duration = animation.get_play_length()
    frames = math.ceil(duration * fps) + 1
    manifest = json.loads((PROJECT / "Saved/demo-state.json").read_text())
    manifest.update(method="Unreal editor animation sampling", animation=animation.get_path_name(),
                    fps=fps, width=width, height=height, shots=list(shots), frames_per_shot=frames,
                    duration_seconds=duration)
    (out / "capture.json").write_text(json.dumps(manifest, indent=2))
    (PROJECT / "Saved/latest-capture.txt").write_text(str(out))
    component.set_animation(animation)
    component.stop()
    component.set_update_animation_in_editor(True)
    world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
    unreal.SystemLibrary.execute_console_command(world, "r.HighResScreenshotDelay 16")
    unreal.SystemLibrary.execute_console_command(world, "r.Shadow.Virtual.SMRT.RayCountLocal 16")
    levels.editor_set_game_view(True)

    def status(value):
        (out / "status.json").write_text(json.dumps({"status": value, "shot": state["shot"],
                                                    "frame": state["frame"], "frames_per_shot": frames}))
    write_status = status

    def tick(dt):
        try:
            if state["task"] is not None:
                if not state["task"].is_task_done():
                    return
                state["task"] = None
                state["frame"] += 1
                state["phase"] = "set"
                if state["frame"] == frames:
                    state["shot"] += 1
                    state["frame"] = 0
                if state["shot"] == len(shots):
                    status("complete")
                    component.set_position(0, False)
                    component.override_animation_data(animation, True, True, 0, 1)
                    component.stop()
                    levels.pilot_level_actor(cameras[shots[0]])
                    stop(False)
                    return
            name = shots[state["shot"]]
            if state["phase"] == "set":
                if state["frame"] == 0:
                    levels.pilot_level_actor(cameras[name])
                    levels.set_exact_camera_view(True)
                # OverrideAnimationData evaluates the pose and refreshes bone transforms.
                # SetPosition alone can leave a paused editor component on a stale pose.
                component.set_position(min(state["frame"] / fps, duration), False)
                component.override_animation_data(animation, False, False,
                                                   min(state["frame"] / fps, duration), 1)
                levels.editor_invalidate_viewports()
                state["ready"] = time.monotonic() + (1.5 if state["frame"] == 0 else .2)
                state["phase"] = "capture"
                status("running")
                return
            if time.monotonic() >= state["ready"]:
                path = out / name / f'{state["frame"]:04d}.png'
                state["task"] = unreal.AutomationLibrary.take_high_res_screenshot(width, height, str(path), delay=0)
        except Exception:
            (out / "error.txt").write_text(traceback.format_exc())
            status("failed")
            stop(False)

    handle = unreal.register_slate_post_tick_callback(tick)
    status("starting")
    unreal.log("Chimera camera capture: " + str(out))
    return str(out)

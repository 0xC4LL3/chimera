"""Small Windows editor controller for the imported skeletal animation."""
import ctypes
import json
import os
import sys
import traceback
from pathlib import Path
import unreal

PROJECT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(PROJECT / "Scripts"))
SAVED = PROJECT / "Saved"
handle = None
component = None
animations = []
clip_index = view_index = 0
held = set()


def play(index=None):
    global clip_index
    if index is not None:
        clip_index = index
    component.play_animation(animations[clip_index], False)
    unreal.log("Chimera play: " + animations[clip_index].get_path_name())


def pause():
    component.stop()


def seek(seconds):
    pause()
    component.set_position(seconds, False)
    component.override_animation_data(animations[clip_index], False, False, seconds, 1)
    unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).editor_invalidate_viewports()


def view(index):
    global view_index
    view_index = index % 3
    name = ("Camera / hero", "Camera / side", "Camera / front")[view_index]
    actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors()
    camera = next(a for a in actors if a.get_actor_label() == name)
    unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).pilot_level_actor(camera)


def status():
    return {"animation": animations[clip_index].get_path_name(), "seconds": component.get_position(),
            "bones": component.get_num_bones(), "camera": view_index}


def stop():
    global handle
    if handle is not None:
        unreal.unregister_slate_post_tick_callback(handle)
        handle = None


def start():
    global handle, component, animations, clip_index, held
    stop()
    state = json.loads((SAVED / "demo-state.json").read_text())
    actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors()
    hero = next(a for a in actors if a.get_actor_label() == "Chimera exported rig")
    component = hero.skeletal_mesh_component
    component.set_update_animation_in_editor(True)
    animations = [unreal.load_asset(path) for path in state["animations"]]
    clip_index = next(i for i, a in enumerate(animations) if a.get_path_name() == state["animation"])
    component.set_animation(animations[clip_index])
    seek(0)
    view(0)
    held = set()
    user32 = ctypes.WinDLL("user32")
    user32.GetForegroundWindow.restype = ctypes.c_void_p
    user32.GetWindowThreadProcessId.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]

    def tick(dt):
        global held
        # The local command file lets the recording operator inspect and cue this editor.
        command = SAVED / "command.py"
        if command.exists():
            code = command.read_text()
            command.unlink()
            try:
                exec(compile(code, str(command), "exec"), globals())
                (SAVED / "command-result.txt").write_text(json.dumps(status()))
            except Exception:
                (SAVED / "command-result.txt").write_text(traceback.format_exc())
        pressed = {key for key in range(0x75, 0x79) if user32.GetAsyncKeyState(key) & 0x8000}
        fresh, held = pressed - held, pressed
        pid = ctypes.c_ulong()
        user32.GetWindowThreadProcessId(user32.GetForegroundWindow(), ctypes.byref(pid))
        if pid.value != os.getpid() or unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).is_in_play_in_editor():
            return
        if 0x75 in fresh: play()
        if 0x76 in fresh: pause()
        if 0x77 in fresh: seek(0)
        if 0x78 in fresh: view(view_index + 1)

    handle = unreal.register_slate_post_tick_callback(tick)
    unreal.EditorPythonScripting.set_keep_python_script_alive(True)
    unreal.log("Chimera preview: F6 replay, F7 pause, F8 reset, F9 camera.")


if __name__ == "__main__":
    start()

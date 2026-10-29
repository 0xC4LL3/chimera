# ComfyUI nodes for Chimera: turn a generated clip into an engine-ready animation of the user's own rig.
# Each node wraps one pipeline script, run as a subprocess of ComfyUI's own Python: scene work
# through Blender's Python API (the bpy package), the rest through torch. A subprocess keeps
# Blender's global state out of ComfyUI. Steps share a run folder, passed as CHIMERA_RUN.
import json, os, shutil, subprocess, sys, tempfile
from pathlib import Path
import numpy as np, torch
from PIL import Image
import folder_paths
import comfy.model_management as mm
from comfy_api.latest import ComfyExtension, InputImpl, IO, Types

HERE = os.path.dirname(os.path.abspath(__file__))
PIPELINE = os.path.join(HERE, "pipeline")
Run = IO.Custom("CHIMERA_RUN")


def call(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"{' '.join(cmd[:4])} failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    return r.stdout


def blender(script, *args):
    # Reference and overlay renders also use the GPU outside ComfyUI's process.
    python(script, *args, vram_gb=2)


def python(script, *args, vram_gb=1):
    # The torch steps run in their own process: free just the GPU memory they need from the models
    # ComfyUI holds (mostly the video model), so the next generation has less to load back.
    mm.free_memory(vram_gb * 2**30, mm.get_torch_device())
    mm.soft_empty_cache()
    return call([sys.executable, os.path.join(PIPELINE, script), *args])


def rig_model(model, out):
    # Rig an unrigged model with SkinTokens; returns the rigged GLB. SkinTokens now and then decodes a
    # malformed skeleton; a new try decodes afresh.
    dst = os.path.join(out, "rigged.glb")
    for attempt in range(3):
        try:
            python("autorig.py", model, dst, vram_gb=12)
            return dst
        except RuntimeError:
            if attempt == 2:
                raise


def run_info(run):
    return json.load(open(os.path.join(run, "run.json")))


def new_run(name):
    base = os.path.join(folder_paths.get_output_directory(), "chimera")
    os.makedirs(base, exist_ok=True)
    return tempfile.mkdtemp(prefix=name + "_", dir=base)


def model_files():
    base = Path(folder_paths.get_input_directory())  # uploads land in 3d/ (editor) or the root (API)
    return sorted(str(p.relative_to(base)).replace("\\", "/") for p in base.rglob("*")
                  if p.suffix.lower() in (".glb", ".gltf", ".fbx"))


class PreviewModel(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ChimeraPreviewModel", display_name="Preview Model (Chimera)", category="chimera",
            description="A GLB of any model Chimera reads (GLB, glTF or FBX), its bone count and animations, and for a rigged model a GLB that shows its bones, for showing it before animating.",
            inputs=[IO.Combo.Input("model_file", options=model_files(), upload=IO.UploadType.model)],
            outputs=[IO.File3DGLB.Output(display_name="glb"), IO.String.Output(display_name="info")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, model_file):
        model = folder_paths.get_annotated_filepath(model_file)
        out = new_run(Path(model).stem + "_preview")
        blender("preview_model.py", model, out)
        info = open(os.path.join(out, "info.json")).read()
        sub = os.path.relpath(out, folder_paths.get_output_directory())
        files = [{"filename": f, "subfolder": sub, "type": "output"}  # API clients only receive files
                 for f in ("preview.glb", "info.json", "skeleton.glb") if os.path.exists(os.path.join(out, f))]
        return IO.NodeOutput(Types.File3D(os.path.join(out, "preview.glb")), info, ui={"text": [info], "files": files})


class AutoRig(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ChimeraAutoRig", display_name="Auto-Rig (Chimera)", category="chimera",
            description="Build a skeleton and skin weights for an unrigged model with SkinTokens, before animating it, and a GLB that shows the new bones.",
            inputs=[IO.Combo.Input("model_file", options=model_files(), upload=IO.UploadType.model)],
            outputs=[IO.File3DGLB.Output(display_name="glb")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, model_file):
        model = folder_paths.get_annotated_filepath(model_file)
        out = new_run(Path(model).stem + "_rig")
        rigged = rig_model(model, out)
        blender("preview_model.py", rigged, out)
        sub = os.path.relpath(out, folder_paths.get_output_directory())
        files = [{"filename": f, "subfolder": sub, "type": "output"} for f in ("rigged.glb", "skeleton.glb", "info.json")]
        return IO.NodeOutput(Types.File3D(rigged), ui={"files": files})


class LoadCreature(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ChimeraLoadCreature", display_name="Load Creature (Chimera)", category="chimera",
            description="Render a rigged creature as the video's first frame: two three-quarter views side by side, or the side view alone.",
            inputs=[
                IO.Combo.Input("model_file", options=model_files(), upload=IO.UploadType.model),
                IO.String.Input("start_action", default="", tooltip="Animation whose first frame is the starting pose; empty for the rest pose."),
                IO.Int.Input("start_frame", default=0, min=0),
                IO.Boolean.Input("grounded", default=True, tooltip="Creature stands on the ground: the capture keeps it above the floor and locks feet that touch it."),
                IO.Boolean.Input("auto_rig", default=False, tooltip="The model has no rig: build a skeleton and skin weights for it with SkinTokens first."),
                IO.Boolean.Input("split_screen", default=False, tooltip="Show two three-quarter views side by side instead of the side view, so the clip shows the motion from two angles (turns toward the camera too). Say so in the prompt."),
            ],
            outputs=[Run.Output(display_name="run"), IO.Image.Output(display_name="reference")],
        )

    @classmethod
    def execute(cls, model_file, start_action, start_frame, grounded, auto_rig=False, split_screen=False):
        model = folder_paths.get_annotated_filepath(model_file)
        run = new_run(Path(model).stem)
        if auto_rig:
            model = rig_model(model, run)
        json.dump({"model": model, "grounded": grounded}, open(os.path.join(run, "run.json"), "w"))
        python("facing.py", model, run, vram_gb=3)
        blender("render_ref.py", model, run, *([start_action, str(start_frame)] if start_action else []), *(["--split"] if split_screen else []))
        blender("dump_rig.py", model, run)
        ref = np.asarray(Image.open(os.path.join(run, "ref.png")).convert("RGB"), dtype=np.float32) / 255
        return IO.NodeOutput(run, torch.from_numpy(ref)[None])


class CaptureMotion(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ChimeraCaptureMotion", display_name="Capture Motion (Chimera)", category="chimera",
            description="Fit the creature's own skeleton to the video: silhouettes and point tracks through the known camera.",
            inputs=[Run.Input("run"), IO.Video.Input("video")],
            outputs=[Run.Output(display_name="run")],
        )

    @classmethod
    def execute(cls, run, video):
        # Captures share the immutable rig arrays. Every take owns its motion and video files.
        creature, run = run, new_run(Path(run).name + "_capture")
        os.link(os.path.join(creature, "rig.npz"), os.path.join(run, "rig.npz"))
        for name in ("run.json", "camera.json", "facing.json", "ref.png"):
            shutil.copy2(os.path.join(creature, name), os.path.join(run, name))
        clip = os.path.join(run, "clip.mp4")
        video.save_to(clip)
        python("track.py", clip, os.path.join(run, "tracks.npz"), os.path.join(run, "camera.json"), vram_gb=9)
        python("fit.py", run, clip, os.path.join(run, "pose.npz"), os.path.join(run, "tracks.npz"),
               *(["--grounded"] if run_info(run)["grounded"] else []), vram_gb=2)
        return IO.NodeOutput(run)


class CleanupMotion(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ChimeraCleanupMotion", display_name="Creature Cleanup (Chimera)", category="chimera",
            description="Smooth the motion and close the loop. Then check how well the result explains the clip.",
            inputs=[Run.Input("run"), IO.Boolean.Input("loop", default=True, tooltip="Make the last frame match the first.")],
            outputs=[Run.Output(display_name="run"), IO.String.Output(display_name="check")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, run, loop):
        python("cleanup.py", run, os.path.join(run, "pose.npz"), os.path.join(run, "clean.npz"), *(["--loop"] if loop else []))
        report = json.loads(python("check.py", run, os.path.join(run, "clean.npz"), os.path.join(run, "clip.mp4")))
        json.dump(report, open(os.path.join(run, "check.json"), "w"))
        if report["ok"]:
            text = f"Check passed: the captured rig matches the clip (overlap {report['mean_overlap']:.2f})."
        elif report["max_turn_deg"] > 45:
            text = (f"Warning: a limb turned {report['max_turn_deg']:.0f} degrees towards the camera, probably to fake "
                    "a shape the rig cannot make. Look at the turned view, or try another clip.")
        else:
            text = (f"Warning: the rig could not follow the clip around frame {report['worst_frame']} (overlap "
                    f"{report['min_overlap']:.2f}). The action may be one this rig cannot make, or seen end-on. "
                    "Try another clip.")
        open(os.path.join(run, "check.txt"), "w").write(text)  # as a file too: API clients only receive files
        sub = os.path.relpath(run, folder_paths.get_output_directory())
        files = [{"filename": f, "subfolder": sub, "type": "output"} for f in ("check.txt", "check.json", "facing.json")]  # facing: for showing the creature's front
        return IO.NodeOutput(run, text, ui={"text": [text], "files": files})


class ExportAnimation(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ChimeraExportAnimation", display_name="Export Animation (Chimera)", category="chimera",
            description="Write the animation onto the original rig as GLB, FBX and BVH, with root-motion and in-place clips.",
            inputs=[Run.Input("run"), IO.String.Input("clip_name", default="attack")],
            outputs=[IO.File3DGLB.Output(display_name="glb"), IO.String.Output(display_name="folder")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, run, clip_name):
        if not clip_name.strip() or clip_name in (".", "..") or any(c in clip_name for c in '/\\\0'):
            raise ValueError("Use a clip name without path separators.")
        out = os.path.join(run, "export")
        blender("export.py", run_info(run)["model"], run, os.path.join(run, "clean.npz"), out, clip_name)
        sub = os.path.relpath(out, folder_paths.get_output_directory())
        files = [{"filename": clip_name + ext, "subfolder": sub, "type": "output"} for ext in (".glb", ".fbx", ".bvh")]
        return IO.NodeOutput(Types.File3D(os.path.join(out, clip_name + ".glb")), out, ui={"files": files})


class Overlay(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ChimeraOverlay", display_name="Rig Overlay (Chimera)", category="chimera",
            description="Video beside the captured rig drawn over it from the same camera, and the rig seen from a turned camera, for checking the fit.",
            inputs=[Run.Input("run"), IO.Boolean.Input("cleaned", default=True)],
            outputs=[IO.Video.Output(display_name="video")],
        )

    @classmethod
    def execute(cls, run, cleaned):
        pose = os.path.join(run, "clean.npz" if cleaned else "pose.npz")
        frames = os.path.join(run, "overlay_frames")
        blender("apply_pose.py", run_info(run)["model"], run, pose, frames)
        out = os.path.join(run, "overlay.mp4")
        python("measure.py", os.path.join(run, "clip.mp4"), frames, out, frames + "_turned")
        return IO.NodeOutput(InputImpl.VideoFromFile(out))


class ChimeraExtension(ComfyExtension):
    async def get_node_list(self):
        return [PreviewModel, AutoRig, LoadCreature, CaptureMotion, CleanupMotion, ExportAnimation, Overlay]


async def comfy_entrypoint():
    return ChimeraExtension()

# ComfyUI nodes for Chimera: turn a generated clip into an engine-ready animation of the user's own rig.
# Each node wraps one pipeline script, run as a subprocess of ComfyUI's own Python: scene work
# through Blender's Python API (the bpy package), the rest through torch. A subprocess keeps
# Blender's global state out of ComfyUI. Steps share a run folder, passed as CHIMERA_RUN.
import json, os, shutil, subprocess, sys, time
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


def blender(script, *args):
    call([sys.executable, os.path.join(PIPELINE, script), *args])


def python(script, *args):
    mm.unload_all_models()  # the torch steps need the GPU memory the video model was holding
    mm.soft_empty_cache()
    call([sys.executable, os.path.join(PIPELINE, script), *args])


def run_info(run):
    return json.load(open(os.path.join(run, "run.json")))


class LoadCreature(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        base = Path(folder_paths.get_input_directory())  # uploads land in 3d/ (editor) or the root (API)
        files = [str(p.relative_to(base)).replace("\\", "/") for p in base.rglob("*")
                 if p.suffix.lower() in (".glb", ".gltf")]
        return IO.Schema(
            node_id="ChimeraLoadCreature", display_name="Load Creature (Chimera)", category="chimera",
            description="Render a rigged creature from a side camera as the video's first frame.",
            inputs=[
                IO.Combo.Input("model_file", options=sorted(files), upload=IO.UploadType.model),
                IO.String.Input("start_action", default="", tooltip="Animation whose first frame is the starting pose; empty for the rest pose."),
                IO.Int.Input("start_frame", default=0, min=0),
            ],
            outputs=[Run.Output(display_name="run"), IO.Image.Output(display_name="reference")],
        )

    @classmethod
    def execute(cls, model_file, start_action, start_frame):
        glb = folder_paths.get_annotated_filepath(model_file)
        run = os.path.join(folder_paths.get_output_directory(), "chimera", f"{Path(glb).stem}_{time.strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(run)
        json.dump({"glb": glb}, open(os.path.join(run, "run.json"), "w"))
        blender("render_ref.py", glb, run, *([start_action, str(start_frame)] if start_action else []))
        blender("dump_rig.py", glb, run)
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
        # ComfyUI caches Load Creature, so every capture of one creature would share its folder;
        # give each capture its own copy of the creature data instead.
        creature, run = run, f"{run}_capture_{time.strftime('%H%M%S')}"
        shutil.copytree(creature, run)
        clip = os.path.join(run, "clip.mp4")
        video.save_to(clip)
        python("track.py", clip, os.path.join(run, "tracks.npz"))
        python("fit.py", run, clip, os.path.join(run, "pose.npz"), os.path.join(run, "tracks.npz"))
        return IO.NodeOutput(run)


class CleanupMotion(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ChimeraCleanupMotion", display_name="Creature Cleanup (Chimera)", category="chimera",
            description="Smooth the motion, close the loop and lock planted feet.",
            inputs=[
                Run.Input("run"),
                IO.Boolean.Input("loop", default=True, tooltip="Make the last frame match the first."),
                IO.Boolean.Input("grounded", default=False, tooltip="Creature stands on the ground: lock feet that touch it in the starting pose."),
            ],
            outputs=[Run.Output(display_name="run")],
        )

    @classmethod
    def execute(cls, run, loop, grounded):
        args = (["--loop"] if loop else []) + (["--contacts", "auto"] if grounded else [])
        python("cleanup.py", run, os.path.join(run, "pose.npz"), os.path.join(run, "clean.npz"), *args)
        return IO.NodeOutput(run)


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
        out = os.path.join(run, "export")
        blender("export.py", run_info(run)["glb"], run, os.path.join(run, "clean.npz"), out, clip_name)
        sub = os.path.relpath(out, folder_paths.get_output_directory())
        files = [{"filename": clip_name + ext, "subfolder": sub, "type": "output"} for ext in (".glb", ".fbx", ".bvh")]
        return IO.NodeOutput(Types.File3D(os.path.join(out, clip_name + ".glb")), out, ui={"files": files})


class Overlay(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ChimeraOverlay", display_name="Rig Overlay (Chimera)", category="chimera",
            description="Video beside the captured rig drawn over it, from the same camera, for checking the fit.",
            inputs=[Run.Input("run"), IO.Boolean.Input("cleaned", default=True)],
            outputs=[IO.Video.Output(display_name="video")],
        )

    @classmethod
    def execute(cls, run, cleaned):
        pose = os.path.join(run, "clean.npz" if cleaned else "pose.npz")
        frames = os.path.join(run, "overlay_frames")
        blender("apply_pose.py", run_info(run)["glb"], run, pose, frames)
        out = os.path.join(run, "overlay.mp4")
        python("measure.py", os.path.join(run, "clip.mp4"), frames, out)
        return IO.NodeOutput(InputImpl.VideoFromFile(out))


class ChimeraExtension(ComfyExtension):
    async def get_node_list(self):
        return [LoadCreature, CaptureMotion, CleanupMotion, ExportAnimation, Overlay]


async def comfy_entrypoint():
    return ChimeraExtension()
